# Copyright (c) Westlake University CAIRI AI Lab.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_dataset, create_loader, AugMixDataset


def build_dataset(args, data_config, collate_fn, num_aug_splits):
    """
    Build training and evaluation datasets and data loaders.
    
    Args:
        args: Training arguments containing dataset configuration
        data_config: Data configuration dictionary
        collate_fn: Collate function for batching
        num_aug_splits: Number of augmentation splits
    
    Returns:
        loader_train: Training data loader
        loader_eval: Evaluation data loader
    """
    if args.dataset.lower() in ['cifar100', 'cifar10']:
        if args.dataset.lower() == 'cifar100':
            DatasetClass = CIFAR100
            data_config['mean'] = (0.5071, 0.4865, 0.4409)
            data_config['std'] = (0.2673, 0.2564, 0.2762)
            assert args.num_classes == 100, "Please check the number of classes."
        else:
            DatasetClass = CIFAR10
            data_config['mean'] = (0.4914, 0.4822, 0.4465)
            data_config['std'] = (0.2023, 0.1994, 0.2010)
            assert args.num_classes == 10, "Please check the number of classes."

        # create datasets
        dataset_train = DatasetClass(root=args.data_dir, train=True, download=args.dataset_download)
        dataset_eval  = DatasetClass(root=args.data_dir, train=False, download=args.dataset_download)

    elif args.dataset.lower() in ['imagenet', 'imagefolder', '']:
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)
        # wrap dataset in AugMix helper
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
    else:
        raise ValueError(f"Do not support the dataset of {args.dataset.lower()}")

    # optional debug subset for sanity checks
    if args.debug_subset and args.debug_subset > 0:
        subset_size = int(args.debug_subset)
        dataset_train = Subset(dataset_train, list(range(min(subset_size, len(dataset_train)))))
        dataset_eval = Subset(dataset_eval, list(range(min(subset_size, len(dataset_eval)))))

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    return loader_train, loader_eval
