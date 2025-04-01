# Python Standard Library
import time
from pathlib import Path

# Third Party Libraries
import numpy as np
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import Compose
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for UNet model.

    Args:
        model (torch.nn.Module): The UNet model.
        loss_fn (callable): Loss function.
        metric (callable): Metric for evaluation.
        data_loader (torch.utils.data.DataLoader): Data loader used to compute test/val metrics.
            Should be the test or validation data loader, depending on run.
        original_files (list): List of original file paths for visualization.
    """
    def __init__(
            self, 
            val_loader, 
            original_files,
            loss_fn=DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3), 
            metric=DiceMetric(include_background=False, reduction="mean_batch", num_classes=2, ignore_empty=False),
            metric_iou=MeanIoU(include_background=False, reduction="mean_batch", ignore_empty=False),
            color_mode='color',
            classes=1,
            visualization_path: Path | None = None
        ):
        super(UNetLightning, self).__init__()
        self.model = init_unet_model(
            in_channels=3 if color_mode == 'color' else 1,
            out_channels=classes
        )
        self.loss_fn = loss_fn
        self.metric_f1 = metric
        self.metric_iou = metric_iou
        self.val_loader = val_loader
        self.original_files = original_files
        self.train_epoch_start = 0
        self.val_epoch_start = 0
        self.color_mode = color_mode
        self.classes = classes
        self.visualization_path = visualization_path
        self.test_outputs_cache = []

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model(x)

    def on_train_epoch_start(self):
        self.train_epoch_start = time.time()

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch (dict): Dictionary containing input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        return loss
    
    def on_train_epoch_end(self):
        self.log(
            "train_epoch_time", time.time() - self.train_epoch_start, 
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch (dict): Dictionary containing input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Dice metric value for the batch.
        """
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        if self.classes == 1:
            val_outputs = torch.sigmoid(outputs) > 0.5
        else:
            val_outputs = torch.softmax(outputs, dim=1)
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            labels = torch.argmax(labels, dim=1, keepdim=True)

        self.metric_f1(val_outputs, labels)
        self.metric_iou(val_outputs, labels)
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def on_validation_epoch_start(self):
        self.val_epoch_start = time.time()

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        dice_tensor = self.metric_f1.aggregate()
        dice = dice_tensor.mean().item() if self.classes > 1 else dice_tensor.item()
        iou_tensor = self.metric_iou.aggregate()
        iou = iou_tensor.mean().item() if self.classes > 1 else iou_tensor.item()

        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_epoch_time", time.time() - self.val_epoch_start, 
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        self.metric_f1.reset()
        self.metric_iou.reset()

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and scheduler configurations.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'frequency': 1,
            }
        }

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)

        if self.classes == 1:
            preds = torch.sigmoid(outputs) > 0.5
        else:
            preds = torch.softmax(outputs, dim=1)
            
            if labels.shape[1] > 1:
                pass
            else:
                labels = torch.nn.functional.one_hot(labels.squeeze(1), num_classes=self.classes).permute(0, 3, 1, 2).float()


        self.metric_f1(preds, labels)
        self.metric_iou(preds, labels)

        # Compute per-image IoU manually for caching
        for i in range(preds.shape[0]):
            pred_i = preds[i]
            label_i = labels[i]

            if self.classes == 1:
                # Binary Dice
                intersection = torch.logical_and(pred_i == 1, label_i == 1).sum().item()
                union = (pred_i == 1).sum().item() + (label_i == 1).sum().item()
                dice = 2 * intersection / union if union > 0 else 0.0

                iou = intersection / torch.logical_or(pred_i == 1, label_i == 1).sum().item()
                iou = iou if not np.isnan(iou) else 0.0
            else:
                # Multiclass Dice: mean over all classes (excluding background)
                dice_list = []
                iou_list = []
                for c in range(1, self.classes):  # skip background
                    pred_c = (pred_i == c)
                    label_c = (label_i == c)

                    inter = torch.logical_and(pred_c, label_c).sum().item()
                    pred_sum = pred_c.sum().item()
                    label_sum = label_c.sum().item()

                    dice_c = (2 * inter) / (pred_sum + label_sum) if (pred_sum + label_sum) > 0 else 0.0
                    union_c = torch.logical_or(pred_c, label_c).sum().item()
                    iou_c = inter / union_c if union_c > 0 else 0.0

                    dice_list.append(dice_c)
                    iou_list.append(iou_c)

                dice = float(np.mean(dice_list))
                iou = float(np.mean(iou_list))

            self.test_outputs_cache.append({
                "iou": iou,
                "dice": dice,
                "img": inputs[i].detach().cpu(),
                "pred": pred_i.detach().cpu(),
                "gt": label_i.detach().cpu()
            })
    
    def on_test_epoch_end(self):
        dice_tensor = self.metric_f1.aggregate()
        if self.classes > 1:
            dice = dice_tensor.mean().item()
            for i, class_dice in enumerate(dice_tensor):
                self.log(f"test_dice_class_{i}", class_dice.item(), on_epoch=True, prog_bar=False, logger=True)
        else:
            dice = dice_tensor.item()

        iou_tensor = self.metric_iou.aggregate()
        iou = iou_tensor.mean().item() if self.classes > 1 else iou_tensor.item()
        if self.classes > 1:
            for i, class_iou in enumerate(iou_tensor):
                self.log(f"test_iou_class_{i}", class_iou.item(), on_epoch=True, prog_bar=False, logger=True)
        
        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.metric_f1.reset()
        self.metric_iou.reset()

        if self.visualization_path:
            sorted_samples = sorted(self.test_outputs_cache, key=lambda x: x["dice"], reverse=True)

            top_samples = sorted_samples[:3]
            bottom_samples = sorted_samples[-3:]

            for idx, sample in enumerate(top_samples):
                self.visualize_test_sample(sample, f"best_{idx+1}.png")

            for idx, sample in enumerate(bottom_samples):
                self.visualize_test_sample(sample, f"worst_{idx+1}.png")

    def visualize_test_sample(self, sample, filename) -> None:
        """
        Visualizes and saves the test sample with the highest IoU score.
        
        Args:
            save_path (str): Path to save the output image.
        """
        img = normalize_img_for_plot(sample["img"])
        gt = sample["gt"].squeeze().cpu().numpy()
        pred = sample["pred"].squeeze().cpu().numpy()

        # Plot input, GT, and prediction
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(gt, cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred, cmap="gray")
        axs[2].set_title(f"Prediction (Dice: {sample['dice']:.2f})")
        axs[2].axis("off")

        fig.savefig(self.visualization_path / filename)
        plt.close(fig)


def init_unet_model(in_channels: int = 1, out_channels: int = 1) -> UNet:
    print(f'In Channels: {in_channels}, Out Channels: {out_channels}')
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,   # 3 for RGB images
        out_channels=out_channels, # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,           # Number of residual units in each layer
        norm=Norm.BATCH,
        dropout=0.1
    )


def insert_transform(compose_obj, new_transform, index):
    # Convert to list
    transform_list = list(compose_obj.transforms)
    # Insert at the desired position
    transform_list.insert(index, new_transform)
    # Wrap it back in a Compose
    return Compose(transform_list)


def normalize_img_for_plot(img: torch.Tensor) -> np.ndarray:
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = img_np.astype(np.float32)

    # Rescale only if needed
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > 1.0 or img_min < 0.0:
        img_np = (img_np - img_min) / (img_max - img_min + 1e-5)

    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
    
    return img_np