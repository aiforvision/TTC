from bolts.lr_scheduler import LinearWarmupCosineAnnealingLR
from loss import (
    SupConLoss, ConSupPrototypeLoss,
)
import torch
import os
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
from lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.aggregation import MeanMetric
from models.utils import find_n_prototypes
import timm


class ContrastiveResNet50(LightningModule):
    def __init__(
        self,
        max_epochs: int = 100,
        output_dim: int = 128,
        lr: float = 1e-3,
        batch_norm: bool = False,
        temperature: float = 0.1,
        supervised: bool = True,
        warmup_epochs: int = 0,
        optimizer_name: str = 'adam',
        batch_size: int = 128,
        calc_cosine_sim: bool = False,
        min_class=0,
        image_net_init: bool = False,
        base_temperature=0.07,
        additional_ckpt_at=[],
        model_name: str = 'resnet50',
        number_of_prototypes=None,
        find_prototypes_optimized=False,
        ratio_supervised_majority=-1,
        **kwargs
    ) -> None:
        super().__init__()
        torch.set_printoptions(threshold=10_000)

        self.output_dim = output_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.supervised = supervised
        self.warmup_epochs = warmup_epochs
        self.optim_choice = optimizer_name
        self.batch_size = batch_size
        self.calc_cosine_sim = calc_cosine_sim
        self.min_class = min_class
        self.number_of_prototypes = number_of_prototypes
        self.find_prototypes_optimized = find_prototypes_optimized
        self.additional_ckpt_at = additional_ckpt_at
        self.model_name = model_name
        self.image_net_init = image_net_init

        self.start_pull = 0.0

        self.save_hyperparameters()

        # Prepare base encoder
        if model_name.startswith('vit'):
            # Use timm to create ViT models
            self.base_encoder = timm.create_model(model_name, pretrained=image_net_init)
            self.num_ftrs = self.base_encoder.embed_dim
        elif model_name == 'resnet50':
            if image_net_init:
                self.base_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.base_encoder = resnet50(weights=None)
            self.num_ftrs = self.base_encoder.fc.in_features
            assert self.num_ftrs == 2048
            # Remove the last layer (classifier) from the resnet
            self.base_encoder = nn.Sequential(*list(self.base_encoder.children())[:-1])
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        # Projection head
        if batch_norm:
            self.projection_head = nn.Sequential(
                nn.Linear(self.num_ftrs, self.num_ftrs, bias=False),
                nn.BatchNorm1d(self.num_ftrs),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_ftrs, output_dim, bias=False),
                nn.BatchNorm1d(output_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.num_ftrs, self.num_ftrs, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_ftrs, output_dim, bias=True),
            )

        # Criterion
        self.critereon = SupConLoss(temperature=self.temperature,
                                    base_temperature=base_temperature,
                                    min_class=self.min_class,
                                    ratio_supervised_majority=ratio_supervised_majority)

        # Metrics
        self.top5_acc_train = torchmetrics.Accuracy(
            task='multiclass', top_k=5, num_classes=2*batch_size-1)
        self.top1_acc_train = torchmetrics.Accuracy(
            task='multiclass', top_k=1, num_classes=2*batch_size-1)

        self.top5_acc_val = torchmetrics.Accuracy(
            task='multiclass', top_k=5, num_classes=2*batch_size-1)
        self.top1_acc_val = torchmetrics.Accuracy(
            task='multiclass', top_k=1, num_classes=2*batch_size-1)

        if self.calc_cosine_sim:
            self.s_0_0_metric_train = MeanMetric()
            self.s_1_1_metric_train = MeanMetric()
            self.s_0_1_metric_train = MeanMetric()

            self.s_0_0_metric_val = MeanMetric()
            self.s_1_1_metric_val = MeanMetric()
            self.s_0_1_metric_val = MeanMetric()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_name.startswith('vit'):
            encoding = self.base_encoder.forward_features(x)
            encoding = self.base_encoder.pool(encoding)
        else:
            encoding = self.base_encoder(x)
        encoding = encoding.view(encoding.size(0), -1)  # Flatten the output
        projection = self.projection_head(encoding)
        projection = F.normalize(projection, dim=1)

        return encoding, projection

    def _cosine_sim(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
        sim_matrix = pairwise_cosine_similarity(embeddings).detach()
        mask_0 = labels == 0
        mask_1 = labels == 1
        if mask_0.sum() == 0 or mask_1.sum() == 0:
            print("No samples of one class in batch")
        s_0_0 = sim_matrix[mask_0][:, mask_0]
        s_1_1 = sim_matrix[mask_1][:, mask_1]
        s_0_1 = sim_matrix[mask_0][:, mask_1]
        s_0_0_mean = s_0_0.mean(dim=1).mean()
        s_0_1_mean = s_0_1.mean(dim=1).mean()
        s_1_1_mean = s_1_1.mean(dim=1).mean()

        return s_0_0_mean.item(), s_0_1_mean.item(), s_1_1_mean.item()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        nviews = len(images) - 1  # last view is for the online classifier
        images_unaugmented = images[-1]
        images = torch.cat(images[:-1], dim=0)
        embeddings, projection = self.forward(images)
        len_images = len(images)
        del images

        if self.calc_cosine_sim:
            s_0_0, s_0_1, s_1_1 = self._cosine_sim(projection,
                                                   torch.cat([labels]*nviews, dim=0).detach().cpu())
            try:
                self.s_0_0_metric_train.update(s_0_0)
                self.s_1_1_metric_train.update(s_1_1)
                self.s_0_1_metric_train.update(s_0_1)
            except RuntimeWarning:
                print("RuntimeWarning")
            self.log(f"s_0_0_metric.train", self.s_0_0_metric_train, on_epoch=True, on_step=False)
            self.log(f"s_0_1_metric.train", self.s_0_1_metric_train, on_epoch=True, on_step=False)
            self.log(f"s_1_1_metric.train", self.s_1_1_metric_train, on_epoch=True, on_step=False)

        del embeddings

        f1, f2 = torch.split(projection, [labels.shape[0], labels.shape[0]], dim=0)
        projection = projection.detach()
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        torch.cuda.empty_cache()
        if self.supervised:
            loss, logits, batch_labels = self.critereon(features, labels)
        else:
            loss, logits, batch_labels = self.critereon(features)

        # Log losses
        if not torch.is_tensor(loss) and len(loss) == 3:
            (loss, loss_min, loss_maj) = loss
            self.log('train.loss_min', loss_min.item())
            self.log('train.loss_maj', loss_maj.item())
        self.log('train.loss', loss.item())

        if len_images == 2 * self.batch_size:
            self.top1_acc_train(logits, batch_labels)
            self.top5_acc_train(logits, batch_labels)
        self.log(f"top1.train", self.top1_acc_train, on_epoch=True, on_step=False)
        self.log(f"top5.train", self.top5_acc_train, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        images, labels = batch
        images = torch.cat(images[:-1], dim=0)
        embeddings, projection = self.forward(images)
        embeddings = embeddings.detach().cpu()

        f1, f2 = torch.split(projection, [labels.shape[0], labels.shape[0]], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if labels.shape[0] != self.batch_size:
            return

        if self.supervised:
            loss, logits, batch_labels = self.critereon(features, labels)
        else:
            loss, logits, batch_labels = self.critereon(features)

        # Log losses
        if not torch.is_tensor(loss) and len(loss) == 3:
            (loss, loss_min, loss_maj) = loss
            self.log('val.loss_min', loss_min)
            self.log('val.loss_maj', loss_maj)
        self.log('val.loss', loss)

        if len(images) == 2 * self.batch_size:
            self.top1_acc_val(logits, batch_labels)
            self.top5_acc_val(logits, batch_labels)
            self.log(f"top1.val", self.top1_acc_val, on_epoch=True, on_step=False)
            self.log(f"top5.val", self.top5_acc_val, on_epoch=True, on_step=False)

        if self.calc_cosine_sim:
            s_0_0, s_0_1, s_1_1 = self._cosine_sim(projection,
                                                   torch.cat([labels]*2, dim=0).detach().cpu())
            try:
                self.s_0_0_metric_val.update(s_0_0)
                self.s_1_1_metric_val.update(s_1_1)
                self.s_0_1_metric_val.update(s_0_1)
            except RuntimeWarning:
                print("RuntimeWarning")
            self.log(f"s_0_0_metric.val", self.s_0_0_metric_val, on_epoch=True, on_step=False)
            self.log(f"s_0_1_metric.val", self.s_0_1_metric_val, on_epoch=True, on_step=False)
            self.log(f"s_1_1_metric.val", self.s_1_1_metric_val, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        lr_decay_rate = 0.1

        if self.optim_choice == 'adam':
            optimizer_main = torch.optim.Adam(
                [
                    {'params': self.base_encoder.parameters()},
                    {'params': self.projection_head.parameters()}
                ], lr=self.lr)
        elif self.optim_choice == 'adamw':
            optimizer_main = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=0.05
            )
        else:
            optimizer_main = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-4)
            print("Using SGD optimizer")

        if self.warmup_epochs > 0:
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer_main,
                self.warmup_epochs,
                self.max_epochs,
                warmup_start_lr=self.lr * 0.1,
                eta_min=self.lr * (lr_decay_rate ** 3),
                last_epoch=-1,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_main, T_max=self.max_epochs, eta_min=self.lr * (lr_decay_rate ** 3), last_epoch=-1)

        return {
            "optimizer": optimizer_main,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _embedding_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

    def _generate_embeddings(self, dataset):
        dataloader = self._embedding_dataloader(dataset)
        embeddings = []
        labels = []

        self.base_encoder.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch

                _, projection = self.forward(inputs[-1].to(self.device))
                projection = projection.view(projection.size(0), -1)
                embeddings.append(projection)
                labels.append(targets)

        self.base_encoder.train()
        return torch.cat(embeddings, dim=0).cpu(), torch.cat(labels, dim=0).cpu()

    def on_train_epoch_start(self):
        if self.current_epoch in self.additional_ckpt_at:
            s_dir = os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                f"additional_ckpt_at_{self.current_epoch}.ckpt"
            )
            self.trainer.save_checkpoint(s_dir)


class ContrastiveResNet50Prototypes(ContrastiveResNet50):
    def __init__(
        self,
        *args,
        simulate_prototypes=False,
        negatives_weight=1.0,
        eps=0.1,
        inverse_prototypes=False,
        eps_0=None,
        eps_1=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.simulate_prototypes = simulate_prototypes
        
        self.critereon = ConSupPrototypeLoss(
            temperature=self.temperature,
            base_temperature=self.base_temperature,
            negatives_weight=negatives_weight,
            eps=eps,
            eps_0=eps_0,
            eps_1=eps_1,
            minority_cls=self.min_class,
            max_epoch=self.max_epochs,
        )

        self._set_prototypes(simulate_prototypes=True)

        self.support_label_acc = torchmetrics.Accuracy(num_classes=2, task="multiclass", top_k=1)
        self.inverse_prototypes = inverse_prototypes

    def on_training_start(self):
        print("Setting prototypes before train")

    def _log_distance_to_prototypes(self, projection, labels, split="train"):
        labels_long = torch.cat([labels, labels], dim=0)

        labels_0 = labels_long == 0
        labels_1 = labels_long == 1

        prototypes_stacked = torch.stack(
            (self.prototype_1, self.prototype_2), dim=0).to(self.device)

        sim = pairwise_cosine_similarity(projection, prototypes_stacked)

        if labels_0.any():
            self.log(f"mean_sim_pr_{split}/cls0_to_proto_0", sim[labels_0, 0].mean(), on_epoch=True, on_step=False)
            pro0_acc_cls0 = sim[:, 0] > sim[:, 1]
            self.log(f"mean_acc_pr_{split}/cls0_to_proto_0",
                     pro0_acc_cls0[labels_0].float().mean(), on_epoch=True, on_step=False)

        if labels_1.any():
            self.log(f"mean_sim_pr_{split}/cls1_to_proto_0", sim[labels_1, 0].mean(), on_epoch=True, on_step=False)
            pro0_acc_cls1 = sim[:, 0] > sim[:, 1]
            self.log(f"mean_acc_pr_{split}/cls1_to_proto_0",
                     pro0_acc_cls1[labels_1].float().mean(), on_epoch=True, on_step=False)

        self.log(f"mean_sim_pr_{split}/all_to_proto_0", sim[:, 0].mean(), on_epoch=True, on_step=False)

    def _set_prototypes(self, simulate_prototypes=False):
        if simulate_prototypes:
            self.prototype_1 = torch.randn(self.output_dim)
            self.prototype_1 = F.normalize(self.prototype_1, dim=0)
            self.prototype_2 = -self.prototype_1
            self.critereon.set_prototypes(torch.stack([self.prototype_1, self.prototype_2], dim=0))
            return

        dm = self.trainer.datamodule
        train_ds = dm.train_dataset

        # generate embeddings
        projections, _ = self._generate_embeddings(train_ds)
        if self.min_class is not None and int(self.min_class) == 0:
            self.prototype_2, self.prototype_1 = find_n_prototypes(2, projections)
        self.prototype_1, self.prototype_2 = find_n_prototypes(2, projections)

        if self.inverse_prototypes:
            self.prototype_1, self.prototype_2 = self.prototype_2, self.prototype_1
        self.critereon.set_prototypes(torch.stack([self.prototype_1, self.prototype_2], dim=0))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch

        if len(labels.shape) > 1:
            labels = labels.squeeze()
        len_images = len(images[0])

        n_contrastive_views = 2
        support_labels = torch.zeros(
            labels.shape[0], 2, dtype=torch.float32)

        support_labels[:, 0] = 1.0 - labels
        support_labels[:, 1] = labels

        images = torch.cat(images[:n_contrastive_views], dim=0)
        _, projection = self.forward(images)

        del images
        torch.cuda.empty_cache()

        if self.calc_cosine_sim:
            s_0_0, s_0_1, s_1_1 = self._cosine_sim(projection,
                                                   torch.cat([labels]*n_contrastive_views, dim=0).detach().cpu())
            self.log(f"s_0_0_metric.train", s_0_0,
                     on_epoch=True, on_step=False)
            self.log(f"s_0_1_metric.train", s_0_1,
                     on_epoch=True, on_step=False)
            self.log(f"s_1_1_metric.train", s_1_1,
                     on_epoch=True, on_step=False)

        f1, f2 = torch.split(
            projection, [labels.shape[0], labels.shape[0]], dim=0)

        # bsz, nviews, pdim
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        torch.cuda.empty_cache()
   
        loss, logits, batch_labels = self.critereon(
                features, support_labels)

        if hasattr(self, 'prototype_1') and hasattr(self, 'prototype_2'):
            self._log_distance_to_prototypes(projection, labels, split="train")

        if not torch.is_tensor(loss) and len(loss) == 3:
            (loss, loss_min, loss_maj) = loss
            self.log('train.loss_min', loss_min.item())
            self.log('train.loss_maj', loss_maj.item())
        self.log('train.loss', loss.item())

        if len_images == 2 * self.batch_size:
            self.top1_acc_train(logits, batch_labels)
            self.top5_acc_train(logits, batch_labels)

        self.log(f"top1.train", self.top1_acc_train,
                 on_epoch=True, on_step=False)
        self.log(f"top5.train", self.top5_acc_train,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        images, labels = batch
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        len_images = len(images[0])

        n_contrastive_views = 2

        support_labels = torch.zeros(
            labels.shape[0], 2, dtype=torch.float32)

        support_labels[:, 0] = 1.0 - labels
        support_labels[:, 1] = labels

        images = torch.cat(images[:n_contrastive_views], dim=0)
        _, projection = self.forward(images)

        del images

        f1, f2 = torch.split(
            projection, [labels.shape[0], labels.shape[0]], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
       
        loss, logits, batch_labels = self.critereon(
                features, support_labels)

        self.log('val.loss', loss)

        if hasattr(self, 'prototype_1') and hasattr(self, 'prototype_2'):
            self._log_distance_to_prototypes(projection, labels, split="val")

        if self.calc_cosine_sim:
            s_0_0, s_0_1, s_1_1 = self._cosine_sim(projection,
                                                   torch.cat([labels]*n_contrastive_views, dim=0).detach().cpu())

            self.log(f"s_0_0_metric.val", s_0_0,
                     on_epoch=True, on_step=False)
            self.log(f"s_0_1_metric.val", s_0_1,
                     on_epoch=True, on_step=False)
            self.log(f"s_1_1_metric.val", s_1_1,
                     on_epoch=True, on_step=False)

        if len_images == 2 * self.batch_size:
            self.top1_acc_val(logits, batch_labels)
            self.top5_acc_val(logits, batch_labels)

        self.log(f"top1.val", self.top1_acc_val,
                 on_epoch=True, on_step=False)
        self.log(f"top5.val", self.top5_acc_val,
                 on_epoch=True, on_step=False)


class ContrastiveViTPrototypes(ContrastiveResNet50Prototypes):
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 *args, **kwargs):
        super().__init__(model_name=model_name, *args, **kwargs)

        self.num_ftrs = self.base_encoder.embed_dim

        self.projection_head = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_ftrs, self.output_dim, bias=True),
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.base_encoder.forward_features(x)
        encoding = self.base_encoder.pool(encoding)
        encoding = encoding.view(encoding.size(0), -1)  # Flatten the output
        projection = self.projection_head(encoding)
        projection = F.normalize(projection, dim=1)

        return encoding, projection


class ContrastiveViT(ContrastiveResNet50):
    def __init__(self, use_weight_decay_scheduler: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_weight_decay_scheduler = use_weight_decay_scheduler

        if self.hparams.batch_norm:
            self.projection_head = nn.Sequential(
                nn.Linear(self.num_ftrs, self.num_ftrs, bias=False),
                nn.BatchNorm1d(self.num_ftrs),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_ftrs, self.output_dim, bias=False),
                nn.BatchNorm1d(self.output_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.num_ftrs, self.num_ftrs, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_ftrs, self.output_dim, bias=True),
            )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.base_encoder.forward_features(x)
        encoding = self.base_encoder.pool(encoding)
        encoding = encoding.view(encoding.size(0), -1)  # Flatten the output
        projection = self.projection_head(encoding)
        projection = F.normalize(projection, dim=1)

        return encoding, projection

