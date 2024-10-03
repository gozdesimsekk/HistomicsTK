import openslide
import histomicstk as htk
import numpy as np
from PIL import Image
import os
import gc



def create_patches_with_openslide(slide_path, patch_size=2624, step_size=2624, output_dir="patches"):
    """
    OpenSlide ile slaytı patch'lere bölme - Optimize edilmiş patch boyutu ve adım boyutu
    """
    # Çıktı dizinini kontrol et ve oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # OpenSlide nesnesini oluştur
    slide = openslide.OpenSlide(slide_path)

    # Slaytın boyutlarını al
    slide_width, slide_height = slide.dimensions
    print(slide.dimensions)
    patches = []
    for y in range(0, slide_height - patch_size + 1, step_size):
        for x in range(0, slide_width - patch_size + 1, step_size):
            # Patch'i al ve numpy dizisine çevir
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert("RGB")
            patch_np = np.array(patch)
            
            # Patch'i kaydet
            patch_img = Image.fromarray(patch_np)
            patch_path = os.path.join(output_dir, f"patch_{y}_{x}.png")
            patch_img.save(patch_path)
            patches.append(patch_path)
    
    return patches

def extract_features_from_patch(patch_path):
    """
    Patch'ten öznitelik çıkarma (CSA)
    """
    # Patch'i aç ve numpy dizisine çevir
    patch = Image.open(patch_path)
    patch_np = np.array(patch)

    # Hücre segmentasyonu ve CSA çıkarımı
    nuclei_segmentation = htk.segmentation.nuclear.detect_nuclei_kofahi(patch_np)
    csa = htk.features.compute_nuclei_features(nuclei_segmentation)

    # CSA'yı hesapla (alanların ortalaması)
    csa_mean = np.mean(csa['Area'])
    print(f"Mean Cell Surface Area (CSA) for {patch_path}: {csa_mean}")

    # Belleği temizle
    del patch_np, nuclei_segmentation, csa
    gc.collect()  # Belleği temizle
    
    return csa_mean

def process_all_patches(slide_path):
    """
    Tüm slayttaki patch'lerden öznitelik çıkarma
    """
    # Patch'leri oluştur
    patches = create_patches_with_openslide(slide_path)

    # Öznitelikleri saklamak için bir liste
    all_csa_values = []

    # Her patch için öznitelik çıkar
    for patch_path in patches:
        csa_value = extract_features_from_patch(patch_path)
        all_csa_values.append(csa_value)
    
    # Tüm patch'ler için CSA ortalamasını hesapla
    overall_csa_mean = np.mean(all_csa_values)
    print(f"Overall Mean CSA for all patches: {overall_csa_mean}")

    return overall_csa_mean

# Örnek kullanım
slide_path = '/Users/gozdesimsek/Desktop/thesis/Thesiscode/HistomicsTK/tests/7316UP-3639.ndpi'
process_all_patches(slide_path)
