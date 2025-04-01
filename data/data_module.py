from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import lightning as pl
from lightning import LightningDataModule
from typing import List, Optional

# Dataset imports
from data.iNatData import INaturalistNClasses
from data.utils import create_subsampled_dataset, ApplyTransform
from data.cardiac import CardiacData


class InMemoryDataset(Dataset):
    """Dataset wrapper that loads and caches data in memory for faster access.
    
    Args:
        dataset: The source dataset to load in memory
        transform: Optional transform to apply to images
        load_in_mem: Whether to load data in memory or just wrap the dataset
    """
    def __init__(self, dataset, transform=None, load_in_mem=True):
        self.transform = transform
        self.dataset = dataset
        self.data = None
        self.new_labels = []
        self.labels_count = [0, 0]

        if load_in_mem:
            self.data = []
            for idx in tqdm(range(len(self.dataset)), desc="Loading data into memory"):
                x = self.dataset[idx]
                img, label = x
                self.labels_count[int(label)] += 1
                self.data.append(x)
        else:
            self.data = dataset
        
    def get_cls_num_list(self):
        """Return list of count per class."""
        return self.labels_count
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        new_labels = self.new_labels[idx] if self.new_labels else -1
        
        if self.transform:
            img = self.transform(img)
        return img, label, new_labels
    
class AggregatedLabels(Dataset):
    """Dataset wrapper that aggregates labels (e.g., binary classification from multi-class).
    
    Args:
        dataset: The source dataset
        transform: Optional transform to apply to images
    """
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset
        self.data = []
        self.labels_count = [0, 0]

        # Filter and transform labels
        for idx in tqdm(range(len(self.dataset)), desc="Processing labels"):
            x = self.dataset[idx]
            img, label = x

            if label == 0 or label == 3:  # Only keep specific labels
                new_label = 0 if label == 0 else 1
                self.labels_count[int(new_label)] += 1
                self.data.append((img, new_label))
        
    def get_cls_num_list(self):
        """Return list of count per class."""
        return self.labels_count
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        if self.transform:
            img = self.transform(img)
        return img, label
    



class SimpleImageDataModule(pl.LightningDataModule):
    """Lightning DataModule for simple image classification datasets.
    
    Args:
        train_dir: Path to training images directory
        val_dir: Path to validation images directory
        test_dir: Path to test images directory
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        train_transform: Transform to apply to training images
        val_transform: Transform to apply to validation/test images
        persistent_workers: Whether to keep worker processes alive between batches
        in_memory: Whether to load data in memory
    """
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform = None,
        val_transform = None,
        persistent_workers = True,
        in_memory = False
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.in_memory = in_memory
        self.transform = train_transform
        self.test_transform = val_transform

    def setup(self, stage: Optional[str] = None):
        """Load datasets. Called by Lightning with the 'fit' and 'test' stages."""
        if self.transform is None:
            raise ValueError("You must provide a transform.")

        self.train_dataset = ImageFolder(
            self.train_dir,
            transform=self.transform
        )
        self.val_dataset = ImageFolder(
            self.val_dir,
            transform=self.transform,
        )
        self.test_dataset = ImageFolder(
            self.test_dir,
            transform=self.test_transform,
        )
        
        if self.in_memory:
            self.train_dataset = InMemoryDataset(self.train_dataset)
            self.val_dataset = InMemoryDataset(self.val_dataset)
            self.test_dataset = InMemoryDataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

class NClassesDataModule(LightningDataModule):
    """Lightning DataModule for N-class datasets (e.g., iNaturalist).
    
    Args:
        data_dir: Path to data directory
        classes: List of class names
        train_transform: Transform to apply to training images
        val_transform: Transform to apply to validation/test images
        num_workers: Number of subprocesses for data loading
        data_set: Dataset name (default: "inat21")
        class_ratios: Optional ratios for class sampling
        batch_size: Number of samples per batch
        seed: Random seed
        subsample_balanced: Whether to balance classes by subsampling
        subsample_upsample: Whether to upsample minority classes
        drop_last: Whether to drop last incomplete batch
        weighte_sampling: Whether to use weighted sampling
        shuffle: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep worker processes alive
        in_memory: Whether to load data in memory
    """
    def __init__(self, data_dir: str, classes: List[str],
                 train_transform=None,
                 val_transform=None,
                 num_workers: int = 32,
                 data_set: str = "inat21",
                 class_ratios: Optional[List[float]] = None,
                 batch_size: int = 64,
                 seed: int = 42,
                 subsample_balanced: bool = False,
                 subsample_upsample: bool = False,
                 drop_last: bool = False,
                 weighte_sampling: bool = False,
                 shuffle: bool = True,
                 pin_memory=True,
                 persistent_workers=True,
                 in_memory=False):

        super().__init__()
        self.data_set = data_set
        self.root_dir = data_dir
        self.classes = classes
        self.class_ratios = class_ratios or []
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.weighte_sampling = weighte_sampling
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.name = f"{self.data_set}_{len(self.classes)}_classes"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.subsample_balanced = subsample_balanced
        self.subsample_upsample = subsample_upsample
        self.persistent_workers = persistent_workers
        self.in_memory = in_memory

        if self.class_ratios and abs(sum(self.class_ratios) - 1.0) > 1e-5:
            raise ValueError("Class ratios must sum to 1.0")

        if subsample_balanced and subsample_upsample:
            raise ValueError(
                "Cannot downsample and subsample to count min, then upsample - they are mutually exclusive")

    def setup(self, stage=None):
        if self.data_set == "inat21":
            self.setup_inat_dataset()
        elif self.data_set == "imagenet":
            raise NotImplementedError(
                "Imagenet dataset setup not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset {self.data_set}")

    def setup_inat_dataset(self):
        """Setup iNaturalist dataset."""
        generator = torch.Generator().manual_seed(self.seed)
        test_dataset = INaturalistNClasses(
            self.root_dir, split="val", transform=self.val_transform, classes=self.classes)
        self.test_dataset, _ = create_subsampled_dataset(
            test_dataset, None, is_test_val=True)

        total_dataset = INaturalistNClasses(
            self.root_dir, split="train", classes=self.classes)
        train_dataset, val_dataset = random_split(
            total_dataset, [0.95, 0.05], generator=generator)

        train_subsampled, train_counts = create_subsampled_dataset(
            train_dataset, self.class_ratios, is_test_val=False,
            subsample_upsample=self.subsample_upsample,
            subsample_balanced=self.subsample_balanced)
        val_subsampled, val_counts = create_subsampled_dataset(
            val_dataset, None, is_test_val=True)

        # Wrap datasets with InMemoryDataset if needed
        if self.in_memory:
            train_subsampled = InMemoryDataset(train_subsampled, transform=None)
            val_subsampled = InMemoryDataset(val_subsampled, transform=None)
            self.test_dataset = InMemoryDataset(self.test_dataset, transform=None)
        
        # Apply transforms
        if self.train_transform is not None:
            self.train_dataset = ApplyTransform(
                train_subsampled, self.train_transform)
            self.val_dataset = ApplyTransform(
                val_subsampled, self.val_transform)
        else:
            self.train_dataset = train_subsampled
            self.val_dataset = val_subsampled

        self.train_counts = [train_counts[cls] for cls in range(len(self.classes))]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory, 
            shuffle=self.shuffle,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=False, 
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )





class CardiacDataModule(LightningDataModule):
    """Lightning DataModule for UKBB cardiac datasets.
    
    Args:
        data_dir: Path to data directory
        train_transform: Transform to apply to training images
        val_transform: Transform to apply to validation/test images
        num_workers: Number of subprocesses for data loading
        minority_class: Minority class name
        batch_size: Number of samples per batch
        seed: Random seed
        drop_last: Whether to drop last incomplete batch
        shuffle: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        use_pil: Whether to use PIL for image loading
        subsample_balanced_train: Whether to balance training data
    """
    def __init__(self, 
                 data_dir: str, 
                 train_transform=None,
                 val_transform=None,
                 num_workers: int = 32,
                 minority_class: str = "cad_broad",
                 batch_size: int = 64,
                 seed: int = 42,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 pin_memory=True,
                 use_pil=False,
                 subsample_balanced_train=False):

        super().__init__()
        self.minority_class = minority_class
        self.root_dir = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.subsample_balanced_train = subsample_balanced_train
        self.use_pil = use_pil
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.test_dataset = CardiacData(
            split="test", 
            transform=self.val_transform, 
            minority_class=self.minority_class, 
            use_pil=self.use_pil
        )
        
        train_dataset = CardiacData(
            split="train", 
            minority_class=self.minority_class, 
            use_pil=self.use_pil
        )
        
        val_dataset = CardiacData(
            split="val",  
            minority_class=self.minority_class, 
            use_pil=self.use_pil
        )
        
        if self.subsample_balanced_train:
            train_dataset, train_counts = create_subsampled_dataset(
                train_dataset, 
                None, 
                subsample_balanced=True, 
                subsample_balanced_percent_of_total=0.05
            )
            self.classes = [0, 1]
        
        # Apply transforms
        self.train_dataset = ApplyTransform(train_dataset, self.train_transform)
        self.val_dataset = ApplyTransform(val_dataset, self.val_transform)
        
        if not self.subsample_balanced_train:
            labels_at_index_0 = [tup[0] for tup in train_dataset.index]
            train_counts = Counter(labels_at_index_0)
            self.classes = list(set(labels_at_index_0))
            
        self.train_counts = [train_counts[cls] for cls in self.classes]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory, 
            shuffle=self.shuffle, 
            persistent_workers=True,
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )


try:
    from medmnist import PneumoniaMNIST, BreastMNIST, ChestMNIST, OCTMNIST
    
    class MedMNISTDataModule(LightningDataModule):
        """Lightning DataModule for MedMNIST datasets.
        
        Args:
            data_dir: Path to data directory
            train_transform: Transform to apply to training images
            val_transform: Transform to apply to validation/test images
            num_workers: Number of subprocesses for data loading
            data_set: Dataset name (pneumonia, breast, chest, oct)
            batch_size: Number of samples per batch
            seed: Random seed
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle training data
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        def __init__(self, data_dir: str, 
                    train_transform=None,
                    val_transform=None,
                    num_workers: int = 32,
                    data_set: str = "",
                    batch_size: int = 64,
                    seed: int = 42,
                    drop_last: bool = True,
                    shuffle: bool = True,
                    pin_memory=True):

            super().__init__()
            self.data_set = data_set
            self.root_dir = data_dir
            self.train_transform = train_transform
            self.val_transform = val_transform
            self.seed = seed
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.pin_memory = pin_memory
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None

        def setup(self, stage=None):
            dataset_mapping = {
                "pneumonia": PneumoniaMNIST,
                "breast": BreastMNIST,
                "chest": ChestMNIST,
                "oct": OCTMNIST
            }

            if self.data_set in dataset_mapping:
                self.setup_dataset(dataset_mapping[self.data_set])
            else:
                raise ValueError(f"Unknown dataset {self.data_set}")

        def setup_dataset(self, dataset_cls):
            """Setup specific MedMNIST dataset.
            
            Args:
                dataset_cls: MedMNIST dataset class to use
            """
            self.test_dataset = self.create_dataset(dataset_cls, 'test')
            self.train_dataset = self.create_dataset(dataset_cls, 'train')
            self.val_dataset = self.create_dataset(dataset_cls, 'val')

            # Handle multi-class datasets by converting to binary
            if dataset_cls == ChestMNIST or dataset_cls == OCTMNIST:
                self.test_dataset = AggregatedLabels(self.test_dataset)
                self.train_dataset = AggregatedLabels(self.train_dataset)
                self.val_dataset = AggregatedLabels(self.val_dataset)

            labels_flattened = self.train_dataset.labels.flatten()
            train_counts = Counter(labels_flattened)
            self.classes = list(set(labels_flattened))
            self.train_counts = [train_counts[cls] for cls in self.classes]

        def create_dataset(self, dataset_cls, split):
            """Create and setup a MedMNIST dataset.
            
            Args:
                dataset_cls: MedMNIST dataset class
                split: Data split (train, val, test)
                
            Returns:
                Configured dataset
            """
            transform = self.train_transform if split == 'train' else self.val_transform
            
            return dataset_cls(
                split=split,
                transform=transform,
                target_transform=None,
                download=True,
                as_rgb=True,
                root=self.root_dir,
                size=224,
            )

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory, 
                shuffle=self.shuffle, 
                persistent_workers=True,
                num_workers=self.num_workers, 
                drop_last=self.drop_last,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=self.shuffle
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )

        def predict_dataloader(self):
            return DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
except ImportError:
    # MedMNIST is optional, provide a placeholder if not available
    class MedMNISTDataModule:
        """Placeholder class when MedMNIST is not available.
        
        This class will raise an error if instantiated.
        """
        def __init__(self, *args, **kwargs):
            raise ImportError("MedMNIST is not installed. Please install with: pip install medmnist")
