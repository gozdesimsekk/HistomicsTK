import collections
import os
import sys
import tempfile

import numpy as np
import packaging.version
import pandas as pd
import skimage.measure
import openslide
print("Openslide import successful")
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

def check_fdata_sanity(fdata, expected_feature_list,
                       prefix='', match_feature_count=True):

    assert len(cfg.nuclei_rprops) == fdata.shape[0]

    if len(prefix) > 0:
        fcols = [col for col in fdata.columns if col.startswith(prefix)]
    else:
        fcols = fdata.columns

    if match_feature_count:
        assert len(fcols) == len(expected_feature_list)

    for col in expected_feature_list:
        assert prefix + col in fcols

class TestFeatureExtraction:

    def test_setup(self):

        # define parameters
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

        # .ndpi dosyasını OpenSlide kullanarak oku
        ndpi_file_path = '/Users/gozdesimsek/Desktop/thesis/Thesiscode/HistomicsTK/tests/7316UP-3639.ndpi'  # ndpi dosya yolu
        slide = openslide.OpenSlide(ndpi_file_path)

        # Görüntüyü oku (küçük çözünürlükte okuma yapabilirsiniz)
        level = 0  # Bu, okuyacağınız çözünürlük seviyesidir. (0 en yüksek çözünürlük)
        im_input = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3]

        # perform color normalization
        im_input_nmzd = htk_cnorm.reinhard(
            im_input, args.reference_mu_lab, args.reference_std_lab)

        # perform color decovolution
        w = htk_cdeconv.rgb_separate_stains_macenko_pca(
            im_input_nmzd, im_input_nmzd.max())

        im_stains = htk_cdeconv.color_deconvolution(im_input_nmzd, w).Stains

        nuclei_channel = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map['hematoxylin'], w)

        im_nuclei_stain = im_stains[:, :, nuclei_channel].astype(float)

        cytoplasm_channel = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map['eosin'], w)

        im_cytoplasm_stain = im_stains[:, :, cytoplasm_channel].astype(
            float)

        # segment nuclei
        im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
            im_nuclei_stain,
            im_nuclei_stain < args.foreground_threshold,
            args.min_radius,
            args.max_radius,
            args.min_nucleus_area,
            args.local_max_search_radius,
        )

        # perform connected component analysis
        nuclei_rprops = skimage.measure.regionprops(im_nuclei_seg_mask)

        # compute nuclei features
        fdata_nuclei = htk_features.compute_nuclei_features(
            im_nuclei_seg_mask, im_nuclei_stain.astype(np.uint8),
            im_cytoplasm=im_cytoplasm_stain.astype(np.uint8))

        cfg.im_input = im_input
        cfg.im_input_nmzd = im_input_nmzd
        cfg.im_nuclei_stain = im_nuclei_stain
        cfg.im_nuclei_seg_mask = im_nuclei_seg_mask
        cfg.nuclei_rprops = nuclei_rprops
        cfg.fdata_nuclei = fdata_nuclei

    def test_compute_features(self):

        # Ortak özelliklerin listesini tutmak için bir dataframe oluştur
        all_features = pd.DataFrame()

        # Intensity features
        intensity_feature_list = [
            'Intensity.Min',
            'Intensity.Max',
            'Intensity.Mean',
            'Intensity.Median',
            'Intensity.MeanMedianDiff',
            'Intensity.Std',
            'Intensity.IQR',
            'Intensity.MAD',
            'Intensity.Skewness',
            'Intensity.Kurtosis',
            'Intensity.HistEnergy',
            'Intensity.HistEntropy',
        ]

        fdata_intensity = htk_features.compute_intensity_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)

        check_fdata_sanity(fdata_intensity, intensity_feature_list)
        all_features = pd.concat([all_features, fdata_intensity], axis=1)

        # Haralick features
        haralick_feature_list = [
            'Haralick.ASM', 'Haralick.Contrast', 'Haralick.Correlation', 'Haralick.SumOfSquares',
            'Haralick.IDM', 'Haralick.SumAverage', 'Haralick.SumVariance', 'Haralick.SumEntropy',
            'Haralick.Entropy', 'Haralick.DifferenceVariance', 'Haralick.DifferenceEntropy',
            'Haralick.IMC1', 'Haralick.IMC2'
        ]

        expected_haralick_list = [f + '.Mean' for f in haralick_feature_list] + \
                                 [f + '.Range' for f in haralick_feature_list]

        fdata_haralick = htk_features.compute_haralick_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain.astype(np.uint8))

        check_fdata_sanity(fdata_haralick, expected_haralick_list)
        all_features = pd.concat([all_features, fdata_haralick], axis=1)

        # Gradient features
        gradient_feature_list = [
            'Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness', 
            'Gradient.Mag.Kurtosis', 'Gradient.Mag.HistEntropy', 
            'Gradient.Mag.HistEnergy', 'Gradient.Canny.Sum', 'Gradient.Canny.Mean'
        ]

        fdata_gradient = htk_features.compute_gradient_features(
            cfg.im_nuclei_seg_mask, cfg.im_nuclei_stain)

        check_fdata_sanity(fdata_gradient, gradient_feature_list)
        all_features = pd.concat([all_features, fdata_gradient], axis=1)

        # Morphometry features
        morphometry_feature_list = [
            'Orientation.Orientation', 'Size.Area', 'Size.ConvexHullArea', 'Size.MajorAxisLength',
            'Size.MinorAxisLength', 'Size.Perimeter', 'Size.Eccentricity', 'Size.Solidity',
            'Size.Extent', 'Shape.FilledArea', 'Shape.EulerNumber', 'Shape.EquivalentDiameter',
            'Shape.Orientation'
        ]

        fdata_morphometry = htk_features.compute_morphometry_features(
            cfg.im_nuclei_seg_mask, cfg.nuclei_rprops)

        check_fdata_sanity(fdata_morphometry, morphometry_feature_list)
        all_features = pd.concat([all_features, fdata_morphometry], axis=1)

        # Tüm özellikleri CSV'ye kaydet
        output_csv = os.path.join(tempfile.gettempdir(), 'combined_features.csv')
        all_features.to_csv(output_csv, index=False)
        print(f'Tüm özellikler {output_csv} dosyasına kaydedildi.')

if __name__ == '__main__':
    test = TestFeatureExtraction()
    test.test_setup()
    test.test_compute_features()
