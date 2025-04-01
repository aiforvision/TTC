from collections import Counter
import os
import os.path
import random
import pickle
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import VisionDataset
from torchvision.transforms.functional import to_pil_image


class CardiacData(VisionDataset):


    def __init__(
        self,
        img_root: str = "" ,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        minority_class: str = "cad_broad",
        use_pil = False,
        #class_ratios: List[float] = None,
    ) -> None:

        self.split = split
        self.minority_class = minority_class
        self.use_pil = use_pil
        self.label_root= f""

        if self.split not in ["train", "val", "test"]:
            raise RuntimeError(f"Unknown split {split}")

        super().__init__(img_root,
                         transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.index: List[Tuple[int,int, torch.Tensor]] = []
        loaded_dict = torch.load(img_root, map_location=torch.device('cpu'))

        
        # load labels
        with open(self.label_root, 'rb') as handle:
            labels = pickle.load(handle)
        
        
        all_items = [(eid, label) for eid, label in labels.items() if eid in loaded_dict]
        
        
        random.seed(42)
        random.shuffle(all_items)

    
        n_val = n_test = int(0.05 * len(all_items))
        n_train = len(all_items) - n_val - n_test

        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train+n_val]
        test_items = all_items[n_train+n_val:]

        if self.split == "train":
            for eid, label in train_items:
                self.index.append((label, eid, loaded_dict[eid]))

        elif self.split == "val":
            for eid, label in val_items:
                self.index.append((label, eid, loaded_dict[eid]))

        elif self.split == "test":
            for eid, label in test_items:
                self.index.append((label, eid, loaded_dict[eid]))



        print(f'Created dataset {self.__class__.__name__} with {len(self)} samples. '\
              f'Classes: {minority_class}.'
              )
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        print(f'Class counts: {cls_counter}')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        class_id, eid, img = self.index[index]

        target: Any = []

        target.append(class_id)

        target = class_id  # int number for the class

        # quick diry fix for all4one
        if self.use_pil:
            img = to_pil_image(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)
    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and os.path.exists(self.label_root)

