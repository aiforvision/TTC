import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from lightning import LightningModule
from bolts.lr_scheduler import LinearWarmupCosineAnnealingLR
import torchmetrics


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reference:
        Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor or float, optional): Class weight factor.
                If a 1D tensor is provided, it must be of size [num_classes].
                If a float is provided, the same alpha is applied to all classes.
                If None, no alpha weighting is applied (i.e., alpha=1 for all classes).
            gamma (float): Exponent of the modulating factor (1 - p_t).
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth labels of shape [batch_size].
        Returns:
            (torch.Tensor): Loss value.
        """
        log_probs = F.log_softmax(inputs, dim=-1)  # [batch_size, num_classes]
        probs = torch.exp(log_probs)               # [batch_size, num_classes]

        # Gather log probabilities and probabilities corresponding to the targets
        log_probs = log_probs.gather(
            dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - probs).pow(self.gamma)
        loss = -focal_term * log_probs

        # Apply alpha if provided
        if self.alpha is not None:
            # If alpha is per-class, index with targets
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(targets.device)[targets]
            else:
                # alpha is a scalar
                alpha_t = self.alpha
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupervisedImageClassifier(LightningModule):
    def __init__(self,
                 max_epochs: int = 100,
                 num_classes: int = 2,
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 dropout: float = 0.0,
                 optimizer_name: str = "adam",
                 warmup_epochs: int = 0,
                 class_weights=None,
                 use_focal_loss: bool = False,
                 focal_gamma: float = 2.0):
        """
        Args:
            max_epochs (int): Number of epochs.
            num_classes (int): Number of classes.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            dropout (float): Dropout rate.
            optimizer_name (str): 'adam' or 'sgd'.
            warmup_epochs (int): Number of warmup epochs.
            class_weights (list or None): Class frequencies or weights to compute normalized class weights.
            use_focal_loss (bool): If True, use Focal Loss instead of CrossEntropyLoss.
            focal_gamma (float): Gamma value for the focal loss.
        """
        super().__init__()

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.num_classes = num_classes
        self.optim_choice = optimizer_name
        self.warmup_epochs = warmup_epochs
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        self.save_hyperparameters()

        # prepare base encoder
        self.base_encoder = resnet50(weights=None)

        # get number of features of last layer
        num_ftrs = self.base_encoder.fc.in_features
        assert num_ftrs == 2048

        # Remove the last layer (classifier) from the resnet
        self.base_encoder = nn.Sequential(
            *list(self.base_encoder.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(num_ftrs, num_classes, bias=True)
        )

        # If class_weights is provided, compute normalized weights
        if class_weights is not None:

            w_0 = 1.0 / class_weights[0]
            w_1 = 1.0 / class_weights[1]
            sum_w = w_0 + w_1

            # Convert to tensor
            class_weights = torch.tensor(
                [w_0 / sum_w, w_1 / sum_w], dtype=torch.float32, device=self.device)

            print(f"Using class weights: {class_weights}")

            if self.use_focal_loss:
                # Use focal loss with per-class alpha
                self.criterion = FocalLoss(
                    alpha=class_weights, gamma=self.focal_gamma, reduction='mean')
                print("Using Focal Loss with class weights")
            else:
                # Standard CrossEntropy with class weights
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # No class weights provided
            if self.use_focal_loss:
                # Use focal loss with alpha=None => alpha=1 for all classes
                self.criterion = FocalLoss(
                    alpha=None, gamma=self.focal_gamma, reduction='mean')
            else:
                self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.val_auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes)

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

        self.test_binf1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        encoding = self.base_encoder(x)
        # [batch_size, 2048, H, W] -> [batch_size, 2048]
        encoding = encoding.view(encoding.size(0), -1)
        return self.classifier(encoding)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        if len(y.shape) > 1:
            y = y.squeeze(1)

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log('train.loss', loss)

        self.train_acc(preds.softmax(dim=-1), y)
        self.log("train.acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1:
            y = y.squeeze(1)
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log('val.loss', loss)

        probs = preds.softmax(dim=-1)
        self.valid_acc(probs, y)
        self.val_auroc(probs, y)

        self.log("val.acc", self.valid_acc)
        self.log('val.auroc', self.val_auroc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log('test.loss', loss)

        probs = preds.softmax(dim=-1)
        self.test_acc(probs, y)
        self.test_auroc(probs, y)
        self.test_binf1(probs, y)

        self.log('test.acc', self.test_acc)
        self.log('test.auc', self.test_auroc)
        self.log('test.binf1', self.test_binf1)

    def configure_optimizers(self):
        lr_decay_rate = 0.1

        if self.optim_choice == 'adam':
            optimizer = torch.optim.Adam(
                [
                    {'params': self.base_encoder.parameters()},
                    {'params': self.classifier.parameters()}
                ],
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-4
            )

        if self.warmup_epochs > 0:
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.warmup_epochs,
                self.max_epochs,
                warmup_start_lr=self.lr * 0.1,
                eta_min=self.lr * (lr_decay_rate ** 3),
                last_epoch=-1,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.lr * (lr_decay_rate ** 3),
                last_epoch=-1
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
