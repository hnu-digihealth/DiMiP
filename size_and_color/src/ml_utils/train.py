# Python Standard Library
from sys import exit
from pathlib import Path

# Third Party Libraries
import numpy as np
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensor, 
    RandRotate90d, RandFlipd, RandAdjustContrastd, 
    RandGaussianNoised, AsDiscreted
)
from monai.data import Dataset, DataLoader
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision.io import read_image
import torchstain

# Local Libraries
from src.ml_utils.preprocessing import HENormalization, ToGrayscale
from src.ml_utils.machine_learning import UNetLightning, insert_transform


def train_and_validate_model(
        train_image_path: Path,
        val_image_path: Path,
        batch_size: int,
        mode: str,
        model_path: Path,
        num_workers: int,
        img_folder: str = 'img',
        lbl_folder: str = 'lbl',
        color_mode: str = 'color',
        normalizer_image_path: Path | None = None,
        classes: int = 1
) -> None:
    # setup data paths
    train_images_path = train_image_path / img_folder
    train_labels_path = train_image_path / lbl_folder
    validate_images_path = val_image_path / img_folder
    validate_labels_path = val_image_path / lbl_folder

    # setup img/label dicts
    train_images = sorted(
        [x for x in train_images_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])
    train_labels = sorted(
        [x for x in train_labels_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])
    validate_images = sorted(
        [x for x in validate_images_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])
    validate_labels = sorted(
        [x for x in validate_labels_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])
    
    train_files = [{'img': img, 'seg': seg} for img, seg in zip(train_images, train_labels)]
    val_files = [{'img': img, 'seg': seg} for img, seg in zip(validate_images, validate_labels)]

    if color_mode == 'color':
        normalizer = torchstain.normalizers.ReinhardNormalizer(method='modified', backend='torch')
        normalizer.fit(read_image(normalizer_image_path))

    # setup transformations
    train_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.int16),
        EnsureChannelFirstd(keys=['img', 'seg']),
        RandAdjustContrastd(keys=['img'], prob=0.5, gamma=(0.7, 1.3)),
        RandGaussianNoised(keys=['img'], prob=0.5, mean=0.0, std=0.01),
        RandFlipd(keys=['img', 'seg'], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=['img', 'seg'], prob=0.5),
        ToTensor(dtype=np.float32),
    ])

    val_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.float32),
        EnsureChannelFirstd(keys=['img', 'seg']),
        ToTensor(dtype=np.float32),
    ])

    if color_mode == 'color':
        train_transforms = insert_transform(
            train_transforms, 
            HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
            2
        )
        val_transforms = insert_transform(
            val_transforms, 
            HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
            2
        )
    else:
        train_transforms = insert_transform(
            train_transforms, 
            ToGrayscale(keys=['img']),
            2
        )
        val_transforms = insert_transform(
            val_transforms,
            ToGrayscale(keys=['img']),
            2
        )

    if classes > 1:
        train_transforms = insert_transform(
            train_transforms, 
            AsDiscreted(keys=["seg"], to_onehot=classes),
            2
        )
        val_transforms = insert_transform(
            val_transforms,
            AsDiscreted(keys=["seg"], to_onehot=classes),
            2
        )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # assert the images are correctly transformed
    try:
        if color_mode == 'color':
            assert(train_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
            assert(val_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
        else:
            assert(train_ds[0]['img'].shape == torch.Size([1, 1024, 1024]))
            assert(val_ds[0]['img'].shape == torch.Size([1, 1024, 1024]))
            
        if classes > 1:
            assert(train_ds[0]['seg'].shape == torch.Size([classes, 1024, 1024]))
            assert(val_ds[0]['seg'].shape == torch.Size([classes, 1024, 1024]))
        else:
            assert(train_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
            assert(val_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
    except AssertionError as e:
        print("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        print(e)
        exit(1)

    # setup data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    ) 

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    # model initialization
    if classes == 1:
        pl_model = UNetLightning(val_loader, val_files, color_mode=color_mode)
    else:
        print('Multi Class Mode')
        pl_model = UNetLightning(
            val_loader, val_files,
            color_mode=color_mode,
            classes=classes,
            loss_fn=DiceCELoss(softmax=True),
        )

    trainer = Trainer(
        max_epochs=1000,
        devices=1, accelerator=mode,
        precision=32,
        callbacks=[
            ModelCheckpoint(monitor='val_dice', mode='max', save_top_k=1, verbose=True),
            EarlyStopping(monitor='val_loss', patience=16, mode='min', verbose=True)
        ],
        default_root_dir=model_path
    )

    # train the model
    trainer.fit(pl_model, train_loader, val_loader)