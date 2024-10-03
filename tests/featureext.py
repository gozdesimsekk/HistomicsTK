import histomicstk as htk
import numpy as np
from PIL import Image

def extract_features_from_patch(patch_path):
    # Patch'i aç ve numpy dizisine çevir
    patch = Image.open(patch_path)
    patch_np = np.array(patch)

    # Hücre segmentasyonu ve CSA çıkarımı
    nuclei_segmentation = htk.segmentation.nuclear.detect_nuclei_kofahi(patch_np)
    csa = htk.features.compute_nuclei_features(nuclei_segmentation)

    # CSA'yı hesapla (alanların ortalaması veya toplamı)
    csa_mean = np.mean(csa['Area'])
    print(f"Mean Cell Surface Area (CSA) for {patch_path}: {csa_mean}")
    
    return csa_mean

# Örnek kullanım
patch_path = 'path/to/your/patch.png'
csa_value = extract_features_from_patch(patch_path)
