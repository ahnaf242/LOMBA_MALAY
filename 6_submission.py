import os
import glob
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, KeepLargestConnectedComponent

TEST_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\volumes" 
SUBMISSION_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\submission"
MODEL_PATH = "best_3d_segresnet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SUBMISSION_DIR, exist_ok=True)

def main():
    print(f"Menyiapkan Jawaban Ujian menggunakan device: {DEVICE}")
    test_files = glob.glob(os.path.join(TEST_DIR, "*.nii.gz"))
    
    if not test_files:
        print(f"Data soal dari juri tidak ditemukan di {TEST_DIR}!")
        return

    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=1, init_filters=16).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    loader = LoadImage(image_only=True)
    channel_first = EnsureChannelFirst()
    scaler = ScaleIntensity()
    
    cleaner = KeepLargestConnectedComponent(applied_labels=[1])

    with torch.no_grad():
        for file_path in tqdm(test_files, desc="Menebak Data Juri"):
            filename = os.path.basename(file_path)

            img_nib = nib.load(file_path)
            original_affine = img_nib.affine

            img_tensor = scaler(channel_first(loader(file_path))).unsqueeze(0).to(DEVICE)

            output = sliding_window_inference(inputs=img_tensor, roi_size=(96, 96, 64), sw_batch_size=4, predictor=model)
            output_mask = (torch.sigmoid(output) > 0.5).float()
            
            output_mask[0] = cleaner(output_mask[0])
            output_mask = output_mask.squeeze().cpu().numpy().astype(np.uint8)

            output_path = os.path.join(SUBMISSION_DIR, filename)
            nib.save(nib.Nifti1Image(output_mask, original_affine), output_path)

    print(f"\nBERES! Semua file siap di-ZIP dan dikumpulkan. Cek folder:\n{SUBMISSION_DIR}")

if __name__ == "__main__":
    main()