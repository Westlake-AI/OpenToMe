import os
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader


def create_dataset(dataset_path, batch_size=32, num_workers=4, input_size=224, mean=None, std=None):
    """
    Create a DataLoader for the ImageNet dataset.
    Args:
        dataset_path (str): Path to the ImageNet dataset.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for data loading.
    Returns:
        DataLoader: A DataLoader for the ImageNet validation dataset.
    """

    mean = mean if mean else [0.485, 0.456, 0.406]
    std = std if std else [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'),
        transform=transform
    )
    if not val_dataset:
        raise ValueError(f"No data found in {os.path.join(dataset_path, 'val')}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return val_loader