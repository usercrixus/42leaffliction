from pathlib import Path
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    has_file_allowed_extension,
)


def default_transform(img_size: int):
    """Apply the same preprocessing as training/validation to one PIL image.

    This mirrors the Resize->CenterCrop->ToTensor->Normalize pipeline used by
    resnet (our model).
    """
    mean = (0.485, 0.456, 0.406)  # resnet (our model) requirement
    std = (0.229, 0.224, 0.225)  # resnet (our model) requirement
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            # Converts a PIL image to a float32 tensor scaled to [0, 1].
            transforms.ToTensor(),
            # Puts inputs on the same scale as the pretrained weights.
            transforms.Normalize(mean, std),
        ]
    )
    return tfm


def build_loaders(
    data_dir: str | Path = "data/images_transformed",
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    name_tail: str | None = "_original",
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    tfms = default_transform(img_size)
    tail = name_tail or None  # allow empty string to mean "no filtering"

    def is_valid_file(path: str) -> bool:
        if not has_file_allowed_extension(path, IMG_EXTENSIONS):
            return False
        if tail is None:
            return True
        return Path(path).stem.endswith(tail)

    # load
    base = datasets.ImageFolder(
        str(data_dir), transform=tfms, is_valid_file=is_valid_file
    )  # expects class_name/imagename.jpg layout
    class_to_idx = base.class_to_idx
    # split
    n_total = len(base)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        base, [n_train, n_val], generator=gen
    )
    # created DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader, class_to_idx
