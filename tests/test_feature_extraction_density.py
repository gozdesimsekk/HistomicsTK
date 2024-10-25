import collections
import os
import pandas as pd
import skimage.io
import skimage.measure
import numpy as np

import histomicstk.features as htk_features
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.nuclear as htk_nuclear

class Cfg:
    def __init__(self):
        self.fdata_nuclei = None
        self.im_nuclei_stain = None
        self.im_nuclei_seg_mask = None
        self.nuclei_rprops = None

cfg = Cfg()

class FeatureExtraction:
    def __init__(self):
        self.intensity_features = None
        self.morphometry_features = None
        self.halalick_features = None
        self.gradient_features = None
        self.fsd_features = None

    def setup_image(self, input_image_file, args):
        im_input = skimage.io.imread(input_image_file)[:, :, :3]
        
        # Color normalization
        im_input_nmzd = htk_cnorm.reinhard(im_input, args.reference_mu_lab, args.reference_std_lab)

        # Perform color deconvolution
        w = htk_cdeconv.rgb_separate_stains_macenko_pca(im_input_nmzd, im_input_nmzd.max())
        im_stains = htk_cdeconv.color_deconvolution(im_input_nmzd, w).Stains

        # Nuclei stain
        nuclei_channel = htk_cdeconv.find_stain_index(htk_cdeconv.stain_color_map['hematoxylin'], w)
        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(float)

        # Cytoplasm stain
        cytoplasm_channel = htk_cdeconv.find_stain_index(htk_cdeconv.stain_color_map['eosin'], w)
        im_cytoplasm_stain = im_stains[:, :, cytoplasm_channel].astype(float)

        # Nuclei segmentation
        im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
            im_nuclei_stain,
            im_nuclei_stain < args.foreground_threshold,
            args.min_radius,
            args.max_radius,
            args.min_nucleus_area,
            args.local_max_search_radius,
        )

        # Connected component analysis
        nuclei_rprops = skimage.measure.regionprops(im_nuclei_seg_mask)

        # Compute nuclei features
        fdata_nuclei = htk_features.compute_nuclei_features(
            im_nuclei_seg_mask, im_nuclei_stain.astype(np.uint8),
            im_cytoplasm=im_cytoplasm_stain.astype(np.uint8)
        )

        # Save results in the global config
        cfg.im_nuclei_stain = im_nuclei_stain
        cfg.im_nuclei_seg_mask = im_nuclei_seg_mask
        cfg.nuclei_rprops = nuclei_rprops
        cfg.fdata_nuclei = fdata_nuclei

    def compute_features(self):
        # Compute all features
        self.intensity_features = htk_features.compute_intensity_features(cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain.astype(np.uint8))
        self.morphometry_features = htk_features.compute_morphometry_features(cfg.im_nuclei_seg_mask)
        self.halalick_features = htk_features.compute_haralick_features(cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain.astype(np.uint8))
        self.gradient_features = htk_features.compute_gradient_features(cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)
        self.fsd_features = htk_features.compute_fsd_features(cfg.im_nuclei_seg_mask, Fs=6)

    def save_features_to_csv(self, output_dir, image_name, folder_name):
        # Combine all features into one dataframe
        all_features = pd.concat([self.intensity_features, 
                                  self.morphometry_features, 
                                  self.halalick_features,
                                  self.gradient_features,
                                  self.fsd_features], 
                                 axis=1)
        
        # Define the output folder for this image
        folder_output_dir = os.path.join(output_dir, folder_name)
        if not os.path.exists(folder_output_dir):
            os.makedirs(folder_output_dir)
        
        # Save the dataframe as a CSV file in the corresponding folder
        output_csv_path = os.path.join(folder_output_dir, f"{image_name}_features.csv")
        all_features.to_csv(output_csv_path, index=True)


    def aggregate_features(self, all_feature_dfs, folder_name):
        # Satır sayısı en az olan CSV'yi bul
        min_rows = min([df.shape[0] for df in all_feature_dfs])
        
        # En küçük satıra sahip CSV dosyasına uygun şekilde diğer dosyaları da rastgele seçip yeniden boyutlandırın
        sampled_dfs = []
        for idx, df in enumerate(all_feature_dfs):
            if df.shape[0] > min_rows:
                sampled_df = df.sample(n=min_rows, random_state=42)  # Rastgele örnekleme
                print(f"CSV {idx+1} - Satır sayısı {df.shape[0]}'dan {min_rows}'a indirildi.")
            else:
                sampled_df = df  # Zaten en küçük satıra sahip olan dosya
                print(f"CSV {idx+1} - Satır sayısı zaten {min_rows}.")
            sampled_dfs.append(sampled_df)

        # Tüm örneklenmiş DataFrame'leri birleştirin
        combined_df = pd.concat(sampled_dfs, axis=0)
        
        # Ortalama öznitelikleri hesaplayın
        mean_features = combined_df.mean(axis=0)
        
        # Ortalamayı bir DataFrame'e dönüştürün ve klasör adını 'ImageID' olarak ekleyin
        mean_features_df = pd.DataFrame(mean_features).T
        mean_features_df.insert(0, 'ImageID', folder_name)
        
        return mean_features_df



def process_all_images(input_dir, output_dir):
    all_feature_dfs = []
    
    # Define parameters
    args = {
        'reference_mu_lab': [8.63234435, -0.11501964, 0.03868433],
        'reference_std_lab': [0.57506023, 0.10403329, 0.01364062],
        'min_radius': 12,
        'max_radius': 30,
        'foreground_threshold': 60,
        'min_nucleus_area': 80,
        'local_max_search_radius': 10,
    }
    args = collections.namedtuple('Parameters', args.keys())(**args)

    # Loop through all subfolders (RadiologyIDs) in the input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        # Only process folders that contain PNG files
        if os.path.isdir(folder_path):
            print(f"Processing folder {folder_name}...")
            all_feature_dfs_for_folder = []
            
            # Loop through all PNG files in the subfolder
            for image_name in os.listdir(folder_path):
                if image_name.endswith(".png"):
                    print(f"Processing {image_name}...")
                    input_image_path = os.path.join(folder_path, image_name)
                    fe = FeatureExtraction()
                    
                    # Setup image and compute features
                    fe.setup_image(input_image_path, args)
                    fe.compute_features()

                    # Save individual image features to CSV in the corresponding folder
                    fe.save_features_to_csv(output_dir, image_name, folder_name)

                    # Add to feature list for aggregation
                    all_feature_dfs_for_folder.append(fe.intensity_features)
                    all_feature_dfs_for_folder.append(fe.morphometry_features)
                    all_feature_dfs_for_folder.append(fe.halalick_features)
                    all_feature_dfs_for_folder.append(fe.gradient_features)
                    all_feature_dfs_for_folder.append(fe.fsd_features)

            # Aggregate all features across images in the folder
            mean_features = fe.aggregate_features(all_feature_dfs_for_folder, folder_name)
            
            # Save mean features to CSV (with features as columns) inside the corresponding folder
            folder_output_dir = os.path.join(output_dir, folder_name)
            if not os.path.exists(folder_output_dir):
                os.makedirs(folder_output_dir)
            
            mean_features.to_csv(os.path.join(folder_output_dir, f"{folder_name}_mean_features.csv"), index=False)

if __name__ == '__main__':
    # Define the input and output directories
    input_dir = './slides'  # Replace with your directory containing folders with PNG files
    output_dir = './output_density'  # Replace with your desired output directory

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all images in the input directory
    process_all_images(input_dir, output_dir)
