import os
import SimpleITK as sitk

folder_dicom = r'C:\Users\Timothy\Documents\LOMBA_MALAY\myocardial-perfusion-scintigraphy-image-database-1.0.0\DICOM'               # Folder tempat file mentah .dcm berada
folder_tujuan = r'C:\Users\Timothy\Documents\LOMBA_MALAY\volumes'      # Folder baru untuk menyimpan Soal (Gambar Asli)

if not os.path.exists(folder_tujuan):
    os.makedirs(folder_tujuan)
    print(f"Folder '{folder_tujuan}' berhasil dibuat!")

daftar_file = [f for f in os.listdir(folder_dicom) if f.endswith('.dcm') or f.endswith('.DCM')]

print(f"Ditemukan {len(daftar_file)} file DICOM. Memulai konversi...\n")

berhasil = 0
for nama_file in daftar_file:
    try:
        jalur_dicom = os.path.join(folder_dicom, nama_file)
        
        gambar_3d = sitk.ReadImage(jalur_dicom)

        nama_baru = nama_file.replace('.dcm', '.nii.gz').replace('.DCM', '.nii.gz')
        jalur_nifti = os.path.join(folder_tujuan, nama_baru)
        
        sitk.WriteImage(gambar_3d, jalur_nifti)
        berhasil += 1
        print(f"[OK] Berhasil: {nama_baru}")
        
    except Exception as e:
        print(f"[GAGAL] Error pada {nama_file}: {e}")

print(f"\nSelesai! {berhasil} dari {len(daftar_file)} file berhasil dikonversi ke NIfTI.")