import openslide
import numpy as np
from PIL import Image
import os
import random

def create_and_save_valid_patches_grid(slide_path, patch_size=1024, num_patches=10, tissue_threshold=0.05, output_dir="slides"):
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    output_dir = os.path.join(output_dir, slide_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slide = openslide.OpenSlide(slide_path)
    slide_width, slide_height = slide.dimensions

    # Patch'ler için grid koordinatları oluştur
    grid_x = list(range(0, slide_width - patch_size + 1, patch_size))
    grid_y = list(range(0, slide_height - patch_size + 1, patch_size))
    grid_positions = [(x, y) for x in grid_x for y in grid_y]
    
    random.shuffle(grid_positions)  # Rastgele seçimi kolaylaştırmak için karıştırın

    patches_saved = 0
    used_positions = set()  # Kaydedilen koordinatları takip etmek için küme

    for (random_x, random_y) in grid_positions:
        if patches_saved >= num_patches:
            break

        # Eğer pozisyon daha önce kullanıldıysa, atla
        if (random_x, random_y) in used_positions:
            continue

        patch = slide.read_region((random_x, random_y), 0, (patch_size, patch_size))
        patch = patch.convert("RGB")
        patch_np = np.array(patch)
        tissue_percentage = calculate_tissue_percentage(patch_np)

        # Yeterli doku yüzdesine sahip yamaları kaydet
        if tissue_percentage > tissue_threshold:
            patch_path = os.path.join(output_dir, f"patch_{random_y}_{random_x}.png")
            Image.fromarray(patch_np).save(patch_path)
            print(f"Saved patch: {patch_path}")
            patches_saved += 1
            used_positions.add((random_x, random_y))  # Kullanılan pozisyonu kaydet

    if patches_saved < num_patches:
        print(f"Only {patches_saved} valid patches were found in grid-based approach.")

def calculate_tissue_percentage(patch_np, threshold=200):
    """
    Patch içinde doku yüzdesini hesaplamak 
    threshold: bir pikselin RGB si 200 den düşükse -> doku içeriyor 
    threshold üstündeki değerler açık ve beyaz renkli kısımları (doku hariç) temsil etmektedir. örneğin RGB(255, 255, 255)
    """
    tissue_pixels = np.sum(np.all(patch_np < threshold, axis=-1))
    total_pixels = patch_np.shape[0] * patch_np.shape[1]
    tissue_percentage = tissue_pixels / total_pixels
    return tissue_percentage

# Kullanım
slide_path = '/Users/gozdesimsek/Desktop/thesis/Thesiscode/HistomicsTK/tests/7316UP-1206.ndpi'
create_and_save_valid_patches_grid(slide_path, patch_size=1024, num_patches=10, tissue_threshold=0.05)
