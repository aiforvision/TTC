from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch, os
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Optimizer
from torchmetrics.functional import accuracy, auroc
import warnings

from bolts.evaluator import SSLEvaluator

class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )

    """

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        dataset: Optional[str] = None,
        calc_auroc: bool = False,
        lr = 1e-3,
    ) -> None:
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None
        self.num_classes: Optional[int] = num_classes
        self.dataset: Optional[str] = dataset
        self.calc_auroc = calc_auroc
        self.lr = lr

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes
        if self.dataset is None:
            self.dataset = trainer.datamodule.name

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel.use_ddp:
                from torch.nn.parallel import DistributedDataParallel

                self.online_evaluator = DistributedDataParallel(self.online_evaluator, device_ids=[pl_module.device])
            elif accel.use_dp:
                from torch.nn.parallel import DataParallel

                self.online_evaluator = DataParallel(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=self.lr)
        print(f"Using Adam optimizer for finetuning {self.online_evaluator.parameters()}")

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == "stl10":
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]

        if len(y.shape) > 1:
            y = y.flatten()
        x = x.to(device)
        y = y.to(device)



        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):  
        if len(batch) == 3:
            # ffcv case
            (images, labels, images2) = batch
            images = [images, images2]
            batch = (images, labels)
        with torch.no_grad(), set_training(pl_module, False):
            x, y = self.to_device(batch, pl_module.device)
            res = pl_module(x)
            if len(res) == 2:
                representations, _ = res
            elif len(res) == 3:
                representations, _, _ = res
            else:
                representations = res
            
            representations = representations.flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        acc = accuracy(mlp_logits.softmax(-1), y, task="multiclass", num_classes=self.num_classes)

        auroc_score = None
        if self.calc_auroc:
            auroc_score = auroc(mlp_logits.softmax(-1), y, task="multiclass", num_classes=self.num_classes)

        return acc,auroc_score, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        try:
            train_acc,auroc_score, mlp_loss = self.shared_step(pl_module, batch)
        except RuntimeWarning:
            print("RuntimeWarning in shared_step")

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

        if self.calc_auroc:
            pl_module.log("online_train_auroc", auroc_score, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        val_acc,auroc_score, mlp_loss = self.shared_step(pl_module, batch)
        pl_module.log("online_val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.calc_auroc:
            pl_module.log("online_val_auroc", auroc_score, on_step=False, on_epoch=True)


    def state_dict(self) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._recovered_callback_state = state_dict

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        if trainer.current_epoch % 10 == 0:
            dicts = self.state_dict()
            
            save_path = f"{trainer.default_root_dir}/checkpoints/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(dicts["state_dict"], os.path.join(save_path, f"epoch={trainer.current_epoch}_ssl_state.pt"))



@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).

    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
