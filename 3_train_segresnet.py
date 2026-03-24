import os
import glob
import torch
from tqdm import tqdm

# Library Medis MONAI
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, ToTensord, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandAffined, RandGaussianNoised, RandAdjustContrastd
)
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

IMAGE_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\volumes"
MASK_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\myocardial-perfusion-scintigraphy-image-database-1.0.0\NIfTI"

BATCH_SIZE = 1 
ACCUMULATION_STEPS = 4  
EPOCHS = 50
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 Memulai persiapan SegResNet di device: {DEVICE}")

    mask_files = glob.glob(os.path.join(MASK_DIR, "*_mask.nii.gz"))
    data_dicts = []
    
    for mask_path in mask_files:
        img_name = os.path.basename(mask_path).replace("_mask", "")
        img_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(img_path):
            data_dicts.append({"image": img_path, "label": mask_path})

    if not data_dicts:
        print(" Data tidak ditemukan!")
        return

    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    print(f"✅ Data Ready! Train: {len(train_files)} pasang | Val: {len(val_files)} pasang")

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 64)),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 64), pos=3, neg=1, num_samples=2,
        ),
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10, rotate_range=0.1, scale_range=0.1, mode=('bilinear', 'nearest')),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 4.5)),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = SegResNet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1, 
        init_filters=16
    ).to(DEVICE)

    loss_function = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    print("\nTRAINING SEGRESNET (50 EPOCH) DIMULAI...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad() 
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for i, batch_data in enumerate(loop):
            inputs, labels = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * ACCUMULATION_STEPS
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        avg_loss = epoch_loss / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
                
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=(96, 96, 64),
                    sw_batch_size=4,
                    predictor=model
                )
                
                val_outputs = [torch.sigmoid(i) > 0.5 for i in val_outputs]
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            scheduler.step(metric) 

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Dice: {metric:.4f}")

            if metric > best_dice:
                best_dice = metric
                torch.save(model.state_dict(), "best_3d_segresnet.pth")
                print(f"REKOR BARU! SegResNet disimpan dengan Dice: {best_dice:.4f}")

    print(f"\nTRAINING SELESAI!")
    print(f" Skor SegResNet tertinggi yang berhasil disimpan: {best_dice:.4f}")

if __name__ == "__main__":
    main()