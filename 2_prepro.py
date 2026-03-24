import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Spacingd, Orientationd, NormalizeIntensityd, 
    RandRotated, RandZoomd, RandGaussianNoised, RandFlipd,
    RandSpatialCropd, CenterSpatialCropd, SpatialPadd, ToTensord
)
from monai.data import Dataset

def siapkan_dataloaders():
    print("=== TAHAP 2: PIPELINE 3D MONAI (CROP & AUGMENTASI) ===")
    
    try:
        df_train = pd.read_csv('split_train.csv')
        df_val = pd.read_csv('split_val.csv')
        df_test = pd.read_csv('split_test.csv')
    except FileNotFoundError:
        print("Error: CSV tidak ditemukan. Pastikan kamu sudah menjalankan '1_mapping.py'!")
        return None, None, None
    
    dict_train = [{"image": r['image_path'], "label": r['label_path']} for _, r in df_train.iterrows()]
    dict_val = [{"image": r['image_path'], "label": r['label_path']} for _, r in df_val.iterrows()]
    dict_test = [{"image": r['image_path'], "label": r['label_path']} for _, r in df_test.iterrows()]
    
    target_spacing = (4.0, 4.0, 4.0) 
    patch_size = (64, 64, 64)        
    
    # 2. PIPELINE TRAIN
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotated(keys=["image", "label"], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.5, mode=("bilinear", "nearest")),
        RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5, mode=("bilinear", "nearest")),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        
        ToTensord(keys=["image", "label"]),
    ])
    
    # 3. PIPELINE VAL & TEST
    val_test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        
        ToTensord(keys=["image", "label"]),
    ])
    
    # 4. BUAT DATASET MONAI
    train_ds = Dataset(data=dict_train, transform=train_transforms)
    val_ds = Dataset(data=dict_val, transform=val_test_transforms)
    test_ds = Dataset(data=dict_test, transform=val_test_transforms)
    
    print(f"✅ PyTorch Datasets Siap! (Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)})")
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = siapkan_dataloaders()
    
    if train_ds is not None and len(train_ds) > 0:
        print("\nSedang memuat gambar uji coba (harap tunggu beberapa detik)...")
        sampel = train_ds[0]
        print(f"Uji Coba Berhasil!")
        print(f"Bentuk Tensor Image : {sampel['image'].shape}")
        print(f"Bentuk Tensor Label : {sampel['label'].shape}")
        
        if torch.isnan(sampel['image']).any():
            print("Peringatan: Ada nilai NaN di dalam tensor gambar!")
        else:
            print("Tensor gambar bersih dari NaN, siap masuk ke AI U-Net!")