from torchvision.transforms import v2 as T
from typing import Tuple, List, Optional, Union
import torch


class ThreeCropTransform:
    """Create three versions of the same image: two with a base transform and one with online transform"""

    def __init__(self, transform, online_transform="train", input_height=224):
        self.transform = transform

        if isinstance(online_transform, str):
            if online_transform == "train":
                self.online_transform = T.Compose([
                    T.RandomResizedCrop(input_height),
                    T.RandomHorizontalFlip(),
                    T.PILToTensor(),
                    T.ConvertImageDtype(dtype=torch.float32),
                ])
            elif online_transform == "train_tensor":
                self.online_transform = T.Compose([
                    T.RandomResizedCrop(input_height),
                    T.RandomHorizontalFlip()
                ])
            elif online_transform == "val_tensor":
                self.online_transform = T.Compose([
                    T.Resize(int(input_height + 0.1 * input_height)),
                    T.CenterCrop(input_height),
                ])
            else:  # Default val transform
                self.online_transform = T.Compose([
                    T.Resize(int(input_height + 0.1 * input_height)),
                    T.CenterCrop(input_height),
                    T.PILToTensor(),
                    T.ConvertImageDtype(dtype=torch.float32),
                ])
        else:
            self.online_transform = online_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.online_transform(x)]


class SimCLRTrainTransform:
    '''
    SimCLR augmentation: random cropping, color distortions, and horizontal flipping
    
    See https://arxiv.org/pdf/2002.05709.pdf appendix A
    '''

    def __init__(self, 
                 strength: float = 1.0,
                 img_height: int = 224,
                 normalize: Optional[Tuple[List[float], List[float]]] = None,
                 color_jitter: List[float] = [0.4, 0.4, 0.4, 0.1]) -> None:
        '''
        Parameters:
        -----------
        strength : float
            Strength of color distortion
        img_height : int
            Height of the image
        normalize : Tuple[List[float], List[float]], optional
            Mean and std of the dataset for normalization
        color_jitter : List[float]
            Parameters for color jittering (brightness, contrast, saturation, hue)
        '''
        self.strength = strength
        self.img_height = img_height

        # Color distortion with adjusted strength
        color_jitter_transform = T.ColorJitter(
            *[strength * item for item in color_jitter])

        self.transforms_list = [
            T.RandomResizedCrop(size=img_height, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(torch.nn.ModuleList([color_jitter_transform]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ConvertImageDtype(dtype=torch.float32),
            T.ToTensor(),
        ]

        if normalize is not None:
            self.set_normalize(normalize)
        else:
            self.transform = T.Compose(self.transforms_list)

    def set_normalize(self, normalize):
        self.transform = T.Compose(
            self.transforms_list + [T.Normalize(*normalize)]
        )
        print(f"Set normalize to {normalize}")

    def __call__(self, image):
        return self.transform(image)


class MedicalGreyScaleTrainTransform(SimCLRTrainTransform):
    '''Transform optimized for medical grayscale images'''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        normalize = kwargs.get("normalize", None)
        img_height = kwargs.get("img_height", 224)

        self.transforms_list = [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomResizedCrop(size=img_height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
        ]

        if normalize is not None:
            self.set_normalize(normalize)
        else:
            self.transform = T.Compose(self.transforms_list)


class CardiacTrainTransform:
    '''Transform optimized for cardiac images'''
    
    def __init__(self, img_height: int = 128) -> None:
        self.img_height = img_height

        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.RandomResizedCrop(size=img_height, scale=(0.2, 1), antialias=True),
            T.ConvertImageDtype(dtype=torch.float32),
        ])

    def __call__(self, image):
        return self.transform(image)


class SimCLRValTransform:
    '''
    Validation transform for SimCLR: resize and center crop
    '''

    def __init__(self,
                 img_height: int = 224,
                 normalize: Optional[Tuple[List[float], List[float]]] = None):
        self.img_height = img_height

        transforms = [
            T.Resize(int(img_height + 0.15 * img_height)),
            T.CenterCrop(img_height),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
        ]
        
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))
        else:
            print("No normalization applied")

        self.transform = T.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)


class SimCLRValTransformCard(SimCLRValTransform):
    '''Validation transform optimized for cardiac images'''

    def __init__(self,
                 img_height: int = 128,
                 normalize: Optional[Tuple[List[float], List[float]]] = None):
        
        # Using parent class init but overriding the transform definition
        super().__init__(img_height, normalize)
        
        self.transform = T.Compose([
            T.Resize(int(img_height + 0.15 * img_height), antialias=True),
            T.CenterCrop(img_height),
            T.ConvertImageDtype(dtype=torch.float32),
        ])


class BaselineTrainTransform:
    '''Basic train transform with random crop and flip'''

    def __init__(self,
                 img_height: int = 224,
                 normalize: Optional[Tuple[List[float], List[float]]] = None):
        self.img_height = img_height

        transforms = [
            T.RandomResizedCrop(size=img_height, scale=(0.08, 1.0), 
                               ratio=(0.75, 1.3333333333333333)),
            T.RandomHorizontalFlip(p=0.5),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float32),
        ]
        
        if normalize is not None:
            transforms.append(T.Normalize(*normalize))

        self.transform = T.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)


class NViewsTransform:
    """Create N augmentations of the same image"""

    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]
