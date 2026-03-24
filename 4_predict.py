import os
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity

IMAGE_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\volumes"
OUTPUT_DIR = r"C:\Users\Timothy\Documents\LOMBA_MALAY\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "best_3d_resunet_pro.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_FILENAME = "1.2.840.4267.32.591718155288413424886041785541718021.nii.gz"  
image_path = os.path.join(IMAGE_DIR, TEST_FILENAME)

def main():
    print(f"Memulai Prediksi untuk: {TEST_FILENAME}")
    
    if not os.path.exists(image_path):
        print(f"File {image_path} tidak ditemukan!")
        return

    model = UNet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        num_res_units=2,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    loader = LoadImage(image_only=True)
    channel_first = EnsureChannelFirst()
    scaler = ScaleIntensity()

    img_nib = nib.load(image_path)
    original_affine = img_nib.affine

    img_tensor = loader(image_path)
    img_tensor = channel_first(img_tensor)
    img_tensor = scaler(img_tensor)
    
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    print("🧠 AI sedang memindai gambar...")
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=img_tensor,
            roi_size=(96, 96, 64),
            sw_batch_size=4,
            predictor=model
        )
        output_mask = (torch.sigmoid(output) > 0.5).float()
    output_mask = output_mask.squeeze().cpu().numpy()

    output_mask = output_mask.astype(np.uint8) 
    
    result_nii = nib.Nifti1Image(output_mask, original_affine)
    output_path = os.path.join(OUTPUT_DIR, f"pred_{TEST_FILENAME}")
    nib.save(result_nii, output_path)

    print(f"Prediksi Selesai! Hasil disimpan di:\n{output_path}")

if __name__ == "__main__":
    main()
