import pandas as pd

# Dosyaları oku
mmc2_data = pd.read_csv('mmc2.csv', sep=';')
histo_data = pd.read_csv('../histopathology_cluster_results_selected_features.csv')

# parent_sample_ids ve ImageID sütunlarına göre birleştir
merged_data = pd.merge(mmc2_data, histo_data, 
                      left_on='parent_sample_ids', 
                      right_on='ImageID', 
                      how='inner')

# Sonuçları kaydet
merged_data.to_csv('cluster_mmc2.csv', index=False)

print("Veriler başarıyla birleştirildi ve 'merged_mmc2_histo.csv' dosyasına kaydedildi.")
print(f"Toplam eşleşen veri sayısı: {len(merged_data)}")