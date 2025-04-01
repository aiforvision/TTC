from bolts.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule

import torchmetrics
import sys, os
sys.path.append(os.path.abspath("/baselinescvpr"))


class LinearClassifier(nn.Module):
    """Simple linear classifier with optional dropout."""
    
    def __init__(self, 
                 input_size: int = 2048,
                 num_classes: int = 2,
                 p_dropout: float = 0.0):
        """
        Args:
            input_size: Feature dimension of input
            num_classes: Number of classes to predict
            p_dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(input_size, num_classes, bias=True)
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """MLP classifier with a single hidden layer and BatchNorm."""
    
    def __init__(self,
                 input_size: int = 2048,
                 hidden_dim_size: int = 2048,
                 num_classes: int = 2,
                 p_dropout: float = 0.0):
        """
        Args:
            input_size: Feature dimension of input
            hidden_dim_size: Hidden layer dimension
            num_classes: Number of classes to predict
            p_dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim_size, bias=False),
            nn.BatchNorm1d(hidden_dim_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(hidden_dim_size, num_classes, bias=True),
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)





class FineTuneClassifier(LightningModule):
    """
    LightningModule for fine-tuning a pre-trained backbone with a classifier head.
    Supports linear or MLP classifiers on top of various backbone architectures.
    """
    
    def __init__(self,
                 base_model: nn.Module = None,
                 num_ftrs: int = 2048,
                 num_classes: int = 2,
                 max_epochs: int = 100,
                 lr: float = 1e-3,
                 nesterov: bool = False,
                 p_dropout: float = 0.0,
                 weight_decay: float = 0,
                 hidden_dim_size: int = None,
                 warmup_epochs: int = 0,
                 optimizer_name: str = "adam",
                 trainable_encoder: bool = False,
                 use_backbone: bool = True,
                 base_model_path: str = None,
                 load_special_encoder: str = None):
        """
        Args:
            base_model: Pre-trained encoder model
            num_ftrs: Feature dimension from encoder
            num_classes: Number of classes to predict
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            nesterov: Whether to use Nesterov momentum (for SGD)
            p_dropout: Dropout probability in classifier
            weight_decay: L2 regularization strength
            hidden_dim_size: If provided, use MLP instead of linear classifier
            warmup_epochs: Number of LR warmup epochs
            optimizer_name: Optimizer to use ("adam" or "sgd")
            trainable_encoder: Whether to train the encoder
            use_backbone: Whether to access the backbone directly
            base_model_path: Path to load base model from
            load_special_encoder: Special encoder type ("bcl" or "sbcl")
        """
        super().__init__()

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.p_dropout = p_dropout
        self.nesterov = nesterov
        self.warmup_epochs = warmup_epochs
        self.optimizer_name = optimizer_name
        self.trainable_encoder = trainable_encoder
        self.use_backbone = use_backbone

        self.base_model = base_model
        
     
        if base_model_path is not None:
            # Load standard model
            self.base_model.load_state_dict(torch.load(base_model_path)['state_dict'])

        # Initialize classifier
        if hidden_dim_size is None:
            self.classifier = LinearClassifier(
                input_size=num_ftrs, 
                num_classes=num_classes, 
                p_dropout=p_dropout
            )
        else:
            self.classifier = MLPClassifier(
                input_size=num_ftrs, 
                hidden_dim_size=hidden_dim_size, 
                num_classes=num_classes, 
                p_dropout=p_dropout
            )

        # Set up metrics
        self.val_auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        self.test_binf1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()

        # Configure base model training mode
        if not self.trainable_encoder:
            if self.use_backbone:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                self.base_model.eval()
            else:
                self.base_model.eval()
                
        else:
            self.base_model.train()

    def forward(self, x) -> torch.Tensor:
        """Extract features and classify."""
        if not self.trainable_encoder:
            with torch.no_grad():
                y_hat = self._extract_features(x)
        else:
            y_hat = self._extract_features(x)
            
        # Flatten features if needed
        y_hat = y_hat.view(y_hat.size(0), -1)
        
        return self.classifier(y_hat)
    
    def _extract_features(self, x) -> torch.Tensor:
        """Extract features from the base model."""
        if self.use_backbone:
            out = self.base_model(x)
        else:
            out = self.base_model(x)
            
        # Handle different return types
        if isinstance(out, dict):
            return out['feats']
        elif isinstance(out, tuple) and len(out) == 2:
            return out[0]
        else:
            return out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        x, y = batch
        if len(y.shape) > 1:
            y = y.flatten()

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        
        # Log metrics
        self.log('train.loss.epoch', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train.loss', loss)
        self.train_acc(preds.softmax(-1), y)
        self.log('train.acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        if len(y.shape) > 1:
            y = y.flatten()

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        probs = preds.softmax(-1)
        
        # Log metrics
        self.log('val.loss', loss, on_step=False, on_epoch=True)
        self.val_acc(probs, y)
        
        # AUROC calculation can have non-deterministic behavior
        prev_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        self.val_auroc(probs, y)
        torch.use_deterministic_algorithms(prev_deterministic_setting, warn_only=True)
        
        self.log('val.acc', self.val_acc)
        self.log('val.auroc', self.val_auroc)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        if len(y.shape) > 1:
            y = y.flatten()

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        probs = preds.softmax(-1)
        
        # Log metrics
        self.log('test.loss', loss)
        self.test_acc(probs, y)
        
        # AUROC and F1 calculation can have non-deterministic behavior
        prev_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        self.test_auroc(probs, y)
        self.test_binf1(probs, y)
        torch.use_deterministic_algorithms(prev_deterministic_setting, warn_only=True)
        
        self.log('test.acc', self.test_acc)
        self.log('test.auc', self.test_auroc)
        self.log('test.binf1', self.test_binf1)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Configure optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.lr,
                nesterov=self.nesterov,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        # Configure learning rate scheduler
        lr_decay_rate = 0.1
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
    
