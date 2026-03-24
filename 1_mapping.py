import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def jalankan_mapping():
    print("=== TAHAP 1: MAPPING & RANDOM SPLIT ===")
    
    folder_gambar = r"C:\Users\Timothy\Documents\LOMBA_MALAY\volumes"
    
    folder_mask = r"C:\Users\Timothy\Documents\LOMBA_MALAY\myocardial-perfusion-scintigraphy-image-database-1.0.0\NIfTI"
    
    print(f"Mencari gambar di: {folder_gambar}")
    
    semua_images = sorted(glob.glob(os.path.join(folder_gambar, "*.nii.gz")))
    
    if not semua_images:
        print("Error: Tidak ada satupun gambar .nii.gz yang ditemukan di path tersebut!")
        print("Tolong pastikan path folder_gambar di atas sudah benar dan filenya ada di dalamnya.")
        return

    data_list = []
    
    for img_path in semua_images:
        p_id = os.path.basename(img_path).replace('.nii.gz', '')
        
        lbl_path = os.path.join(folder_mask, p_id + "_mask.nii.gz")
        
        if os.path.exists(lbl_path):
            data_list.append({
                "patient_id": p_id, 
                "image_path": img_path, 
                "label_path": lbl_path
            })

    if len(data_list) == 0:
        print("Error: Gambar ditemukan, TAPI tidak ada satupun Mask/Label pasangannya yang cocok!")
        print("Tolong cek apakah folder_mask sudah benar, atau apakah akhiran nama file mask-nya menggunakan '_mask'?")
        return
    
    df = pd.DataFrame(data_list)
    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    
    df_trainval, df_test = train_test_split(df_unique, test_size=0.20, random_state=42)
    
    df_train, df_val = train_test_split(df_trainval, test_size=0.125, random_state=42)
    
    print(f"\nBerhasil menemukan {len(df_unique)} pasien yang memiliki gambar & mask!")
    print(f"Distribusi Pasien: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    train_data = df[df['patient_id'].isin(df_train['patient_id'])]
    val_data = df[df['patient_id'].isin(df_val['patient_id'])]
    test_data = df[df['patient_id'].isin(df_test['patient_id'])]

    train_data.to_csv('split_train.csv', index=False)
    val_data.to_csv('split_val.csv', index=False)
    test_data.to_csv('split_test.csv', index=False)
    
    print("File CSV Split (Train, Val, Test) berhasil dibuat di folder project-mu.")

if __name__ == "__main__":
    jalankan_mapping()