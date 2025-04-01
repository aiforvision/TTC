from collections import Counter
import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets import VisionDataset


class INaturalistNClasses(VisionDataset):
    """
    Adaptation of the PyTorch INaturalist dataset implementation that contains only
    data from a specific taxonomy subtree (new classes).

    Args:
        root (str): Root directory path containing the dataset
        split (str, optional): The dataset split, either "train" or "val". Default: "train"
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. Default: None
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. Default: None
        classes (List[str], optional): List of class names to include. If None, uses all classes.
            Default: None

    Note:
        This dataset uses the validation set as test set according to the paper 
        "When Does Contrastive Visual Representation Learning Work?"
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.split = split
        self.classes = classes
        
        if self.split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        super().__init__(os.path.join(root, split),
                         transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Check the root path.")

        # Map category id to full path name
        self.all_categories: List[str] = []
        try:
            self.all_categories = sorted(os.listdir(self.root))
        except FileNotFoundError:
            raise RuntimeError(f"Directory not found: {self.root}")

        # Index of all files: (class_id, category_id, filepath)
        self.index: List[Tuple[int, int, str]] = []

        if classes is None:
            # Use full dataset
            print("Using full dataset!")
            self.classes = [cat.split("_", 1)[1].lower() for cat in self.all_categories]
            for dir_index, dir_name in enumerate(self.all_categories):
                self._add_category_to_index(dir_index, dir_index)
        else:
            # Only add samples from specified classes
            for cls_id, cls in enumerate(classes):
                categories_for_cls = self._get_categories_for_class(cls)
                for cat_id in categories_for_cls:
                    self._add_category_to_index(cls_id, cat_id)

        self._print_dataset_info()

    def _add_category_to_index(self, cls_id: int, cat_id: int) -> None:
        """Add all images from a category to the index."""
        cat_path = os.path.join(self.root, self.all_categories[cat_id])
        try:
            files = os.listdir(cat_path)
            for fname in files:
                self.index.append((cls_id, cat_id, fname))
        except FileNotFoundError:
            print(f"Warning: Category directory not found: {cat_path}")

    def _print_dataset_info(self) -> None:
        """Print dataset information and class statistics."""
        print(f'Created dataset {self.__class__.__name__} with {len(self)} samples.')
        print(f'Split: {self.split}')
        print(f'Classes: {self.classes}')
        
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        if self.classes:
            cls_counts = [cls_counter.get(i, 0) for i in range(len(self.classes))]
            print(f'Class counts: {cls_counts}')

    def _get_categories_for_class(self, cls: str) -> List[int]:
        """
        Returns all category IDs that are in the subtree of the given class.
        
        Args:
            cls (str): The class name to search for
            
        Returns:
            List[int]: List of category IDs
        """
        cats: List[int] = []
        cls_lower = cls.lower()
        
        for dir_name in self.all_categories:
            # Remove the numeric prefix
            cat_name = dir_name.split("_", 1)[1].lower()
            if cat_name.startswith(cls_lower):
                cats.append(int(dir_name.split("_", 1)[0]))

        return cats

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get item at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        class_id, cat_id, fname = self.index[index]
        img_path = os.path.join(self.root, self.all_categories[cat_id], fname)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        target = class_id  # int number for the class

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def _check_integrity(self) -> bool:
        """Check if the dataset directory exists and is not empty."""
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0
