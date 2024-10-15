import openslide
import numpy as np
from PIL import Image
import os
import random

def create_and_save_valid_patches(slide_path, patch_size=1024, step_size=1024, num_patches=10, tissue_threshold=0.05, output_dir="slides"):
    """
    OpenSlide ile slaytı patch'lere bölme ve yalnızca %5'ten fazla doku içeren rastgele patch'leri kaydetme
    """
    # Slide'ın dosya adını almak (uzantı hariç)
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    # Çıktı dizinini, slide'ın adıyla aynı olacak şekilde oluştur
    output_dir = os.path.join(output_dir, slide_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide = openslide.OpenSlide(slide_path)
    slide_width, slide_height = slide.dimensions

    # Num_patches kadar geçerli patch bulmaya çalış
    patches_saved = 0
    attempts = 0  # Rastgele seçim sayısını izleyelim
    while patches_saved < num_patches and attempts < num_patches * 10:
        # Rastgele bir koordinat seç
        random_x = random.randint(0, slide_width - patch_size)
        random_y = random.randint(0, slide_height - patch_size)
        
        # Patch'i al
        patch = slide.read_region((random_x, random_y), 0, (patch_size, patch_size))
        patch = patch.convert("RGB")
        patch_np = np.array(patch)

        # Doku yüzdesini hesapla
        tissue_percentage = calculate_tissue_percentage(patch_np)

        # Eğer doku yüzdesi eşik değerinden büyükse kaydet
        if tissue_percentage > tissue_threshold:
            patch_path = os.path.join(output_dir, f"patch_{random_y}_{random_x}.png")
            Image.fromarray(patch_np).save(patch_path)
            print(f"Saved patch: {patch_path}")

            patches_saved += 1  # Başarıyla kaydedilen patch sayısını artır
        attempts += 1  # Rastgele seçimlerin sayısını artır

    if patches_saved < num_patches:
        print(f"Only {patches_saved} valid patches were found after {attempts} attempts.")

def calculate_tissue_percentage(patch_np, threshold=200):
    """
    Patch içinde doku yüzdesini hesapla
    """
    tissue_pixels = np.sum(np.all(patch_np < threshold, axis=-1))
    total_pixels = patch_np.shape[0] * patch_np.shape[1]
    tissue_percentage = tissue_pixels / total_pixels
    return tissue_percentage

# Kullanım
slide_path = '/Users/gozdesimsek/Desktop/thesis/Thesiscode/HistomicsTK/tests/7316UP-3639.ndpi'
create_and_save_valid_patches(slide_path, patch_size=1024, step_size=1024, num_patches=10, tissue_threshold=0.05)
