import os
import glob
import torch
import pandas as pd
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, ToTensord, KeepLargestConnectedComponent
)
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

IMAGE_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\volumes"
MASK_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\myocardial-perfusion-scintigraphy-image-database-1.0.0\NIfTI"
MODEL_PATH = "best_3d_resunet_pro.pth"
OUTPUT_CSV = "hasil_evaluasi_lomba.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("📈 Menyiapkan Evaluasi Massal...")

    mask_files = glob.glob(os.path.join(MASK_DIR, "*_mask.nii.gz"))
    data_dicts = []
    
    for mask_path in mask_files:
        img_name = os.path.basename(mask_path).replace("_mask", "")
        img_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(img_path):
            data_dicts.append({"image": img_path, "label": mask_path, "name": img_name})

    _, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    print(f"📋 Ditemukan {len(val_files)} pasien untuk diuji.")

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="none")
    cleaner = KeepLargestConnectedComponent(applied_labels=[1])

    results = []

    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader, desc="Mengevaluasi")):
            val_inputs = val_data["image"].to(DEVICE)
            val_labels = val_data["label"].to(DEVICE)
            patient_name = val_data["name"][0] # Ambil nama file

            val_outputs = sliding_window_inference(
                inputs=val_inputs, roi_size=(96, 96, 64), 
                sw_batch_size=4, predictor=model
            )

            val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()

            val_outputs[0] = cleaner(val_outputs[0])

            dice_metric(y_pred=val_outputs, y=val_labels)

            score = dice_metric.get_buffer()[-1].item()
            results.append({"Nama Pasien": patient_name, "Dice Score": score})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    avg_dice = df["Dice Score"].mean()
    print("\n==========================================")
    print(f"EVALUASI SELESAI!")
    print(f"Rata-rata Skor Dice Keseluruhan: {avg_dice:.4f} ({(avg_dice*100):.2f}%)")
    print(f"File laporan hasil disimpan di: {OUTPUT_CSV}")
    print("==========================================")

if __name__ == "__main__":
    main()
