"""Image and data transforms for v8 training."""
import torch
import torchvision.transforms as T
from typing import Optional


def get_train_transforms(
    image_size: tuple = (224, 224),
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.2,
    hue: float = 0.05,
    noise_std: float = 0.01,
) -> T.Compose:
    return T.Compose([
        T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        T.RandomApply([AddGaussianNoise(std=noise_std)], p=0.5),
    ])


def get_val_transforms(image_size: tuple = (224, 224)) -> T.Compose:
    return T.Compose([])  # No augmentation for validation


class AddGaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std
