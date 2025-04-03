import pandas as pd
import os

# Dosya yolları - kendi sisteminize göre güncelleyin
base_path = '/Users/gozdesimsek/Desktop/thesis/Thesiscode/HistomicsTK/'
clinical_path = os.path.join(base_path, 'thesis_codes_deneme/histo_mr.csv')
features_path = os.path.join(base_path, 'selected_features.csv')
output_path = os.path.join(base_path, 'merged_clinical_radiomic_data.csv')

# Verileri yükle
clinical_df = pd.read_csv(clinical_path)
features_df = pd.read_csv(features_path)

# Verileri birleştir (sadece eşleşen ImageID'ler)
merged_df = pd.merge(clinical_df, features_df, on='ImageID', how='inner')

# Temizlik ve veri işleme
merged_df = merged_df[
    (~merged_df['Survival_from_surgery_days_UPDATED'].isna()) &
    (merged_df['Survival_from_surgery_days_UPDATED'] != 'Not Available')
].copy()

merged_df['Survival_from_surgery_days_UPDATED'] = pd.to_numeric(
    merged_df['Survival_from_surgery_days_UPDATED'], errors='coerce'
)

# Eksik veri yönetimi
merged_df['MGMT'] = merged_df['MGMT'].fillna('Unknown')
merged_df['IDH1'] = merged_df['IDH1'].fillna('Unknown')

# Eksik veri kontrolü
missing_in_clinical = features_df[~features_df['ImageID'].isin(merged_df['ImageID'])]['ImageID'].tolist()

# Sonuçları göster
print(f"Toplam eşleşen hasta sayısı: {merged_df['PatientID'].nunique()}")
print(f"Toplam eşleşen görüntü sayısı: {len(merged_df)}")
print(f"Klinik veride eksik olan ImageID'ler ({len(missing_in_clinical)} adet): {missing_in_clinical}")

# Benzersiz hasta ve görüntü listeleri
print("\nEşleşen PatientID'ler:", merged_df['PatientID'].unique().tolist())
print("Eşleşen ImageID'ler:", merged_df['ImageID'].unique().tolist())

# Veriyi kaydet
merged_df.to_csv(output_path, index=False)
print(f"\nBirleştirilmiş veri kaydedildi: {output_path}") 