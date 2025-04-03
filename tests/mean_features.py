import pandas as pd
import os
import glob

# output_density klasörünün yolunu belirtin
base_dir = 'output_density'

# Tüm CSV dosyalarını birleştirmek için boş bir liste oluşturun
all_data = []
found_files = False
total_folders = 0
mean_features_count = 0

# Her klasör için _mean_features.csv kontrolü yapmak için sözlük
folders_with_mean_features = {}

# Alt klasörlerdeki tüm {id}_mean_features.csv dosyalarını bulmak için os.walk kullanın
for root, dirs, files in os.walk(base_dir):
    if root != base_dir:  # Ana klasörü saymamak için
        total_folders += 1
    has_mean_features = False
    
    for file in files:
        if file.endswith('_mean_features.csv'):
            found_files = True
            mean_features_count += 1
            has_mean_features = True
            file_path = os.path.join(root, file)
            # CSV dosyasını okuyun ve all_data listesine ekleyin
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    # Her klasör için sonucu kaydet
    if root != base_dir:  # Ana klasörü hariç tut
        folders_with_mean_features[root] = has_mean_features

print(f"output_density klasöründe toplam {total_folders} alt klasör bulunmaktadır.")
print(f"Bunlardan {mean_features_count} tanesi '_mean_features.csv' dosyasına sahiptir.")

print("\n_mean_features.csv dosyası bulunmayan klasörler:")
for folder, has_features in folders_with_mean_features.items():
    if not has_features:
        print(f"- {folder}")

if not found_files:
    print("\nUyarı: Alt klasörlerde hiç '_mean_features.csv' dosyası bulunamadı!")
else:
    # Tüm verileri tek bir DataFrame'de birleştirin
    combined_data = pd.concat(all_data, ignore_index=True)

    # Sonuçları tek bir CSV dosyasına yazın
    combined_data.to_csv('all_density_features_histopathology.csv', index=False)
    print("\nTüm dosyalar başarıyla combined_features.csv dosyasına toplandı.")
