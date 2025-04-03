import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Veriyi oku
data = pd.read_csv('cluster_genetic.csv')

# Matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

# Renk paleti
colors = sns.color_palette("husl", 8)

# Analiz edilecek kategorik değişkenler
categorical_vars = ['who_tumor_grade', 'tumor_location', 'clinical_mgmt_methyl_status', 
                   'mgmt_methyl_stp27_prediction', 'is_gcimp', 'chr10_del', 'chr7_amp',
                   'codel_1p_19q', 'CDKN2A', 'EGFR', 
                   'NF1', 'PDGFRA', 'PIK3CA', 'PIK3R1', 'PTEN', 'RB1', 'TP53', 'TERTp_mut']

for var in categorical_vars:
    # Geçici bir veri kopyası oluştur
    temp_data = data.copy()
    
    # NaN değerleri "Not Available" olarak değiştir
    temp_data[var] = temp_data[var].fillna('Not Available')
    
    # Tüm verileri içeren crosstab
    ct_all = pd.crosstab(temp_data['Cluster'], temp_data[var])
    
    # Görselleştirme (tüm verilerle)
    plt.figure()
    ax = ct_all.plot(kind='bar', stacked=True, color=colors[:len(ct_all.columns)])
    plt.title(f'Cluster\'lara Göre {var} Dağılımı (Tüm Veriler)', pad=20)
    plt.xlabel('Cluster')
    plt.ylabel('Hasta Sayısı')
    plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'genetichisto/cluster_{var}_distribution_all.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Not Available ve Not Performed olmayan verileri filtrele
    valid_data = temp_data[~temp_data[var].isin(['Not Available', 'Not Performed', 'No alteration'])]
    
    if len(valid_data) > 0:
        # Geçerli verilerle crosstab
        ct_valid = pd.crosstab(valid_data['Cluster'], valid_data[var])
        
        # Görselleştirme (sadece geçerli verilerle)
        plt.figure()
        ax = ct_valid.plot(kind='bar', stacked=True, color=colors[:len(ct_valid.columns)])
        plt.title(f'Cluster\'lara Göre {var} Dağılımı (Sadece Geçerli Veriler)', pad=20)
        plt.xlabel('Cluster')
        plt.ylabel('Hasta Sayısı')
        plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'genetichisto/cluster_{var}_distribution_valid.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Yüzdesel dağılım görselleştirmesi
        plt.figure()
        percentage_data = (ct_valid.div(ct_valid.sum(axis=1), axis=0) * 100)
        ax = percentage_data.plot(kind='bar', stacked=True, color=colors[:len(ct_valid.columns)])
        plt.title(f'Cluster\'lara Göre {var} Yüzdesel Dağılımı (Sadece Geçerli Veriler)', pad=20)
        plt.xlabel('Cluster')
        plt.ylabel('Yüzde (%)')
        plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
        plt.tight_layout()
        plt.savefig(f'genetichisto/cluster_{var}_percentage_valid.png', bbox_inches='tight', dpi=300)
        plt.close()

# İstatistiksel analiz sonuçları
chi_square_results = {'all_data': {}, 'valid_data': {}}

for var in categorical_vars:
    # Geçici veri kopyası
    temp_data = data.copy()
    temp_data[var] = temp_data[var].fillna('Not Available')
    
    if var == 'who_tumor_grade':
        temp_data[var] = temp_data[var].astype(str)
        temp_data[var] = temp_data[var].replace('nan', 'Not Available')
    
    # Tüm verilerle analiz
    contingency_table_all = pd.crosstab(temp_data['Cluster'], temp_data[var])
    chi2_all, p_value_all, dof_all, expected_all = stats.chi2_contingency(contingency_table_all)
    chi_square_results['all_data'][var] = {
        'chi2': chi2_all,
        'p_value': p_value_all,
        'dof': dof_all
    }
    
    # Sadece geçerli verilerle analiz
    valid_data = temp_data[~temp_data[var].isin(['Not Available', 'Not Performed', 'No alteration'])]
    if len(valid_data) > 0:
        contingency_table_valid = pd.crosstab(valid_data['Cluster'], valid_data[var])
        chi2_valid, p_value_valid, dof_valid, expected_valid = stats.chi2_contingency(contingency_table_valid)
        chi_square_results['valid_data'][var] = {
            'chi2': chi2_valid,
            'p_value': p_value_valid,
            'dof': dof_valid
        }

# Sonuçları yazdır
with open('genetichisto/cluster_genetic_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write('Genetik Değişkenlerin Cluster\'larla İlişkisi - İstatistiksel Analiz\n')
    f.write('=========================================================\n\n')
    
    f.write('\nTÜM VERİLERLE ANALİZ\n')
    f.write('===================\n')
    for var, results in chi_square_results['all_data'].items():
        f.write(f'\n{var} Analizi (Tüm Veriler):\n')
        f.write('-' * (len(var) + 20) + '\n')
        f.write(f'Chi-square değeri: {results["chi2"]:.4f}\n')
        f.write(f'Serbestlik derecesi: {results["dof"]}\n')
        f.write(f'p-değeri: {results["p_value"]:.4f}\n')
        if results["p_value"] < 0.05:
            f.write('Yorum: İstatistiksel olarak anlamlı ilişki bulundu (p < 0.05)\n')
        else:
            f.write('Yorum: İstatistiksel olarak anlamlı ilişki bulunamadı (p >= 0.05)\n')
        f.write('\n')
    
    f.write('\nSADECE GEÇERLİ VERİLERLE ANALİZ\n')
    f.write('==============================\n')
    for var, results in chi_square_results['valid_data'].items():
        f.write(f'\n{var} Analizi (Geçerli Veriler):\n')
        f.write('-' * (len(var) + 23) + '\n')
        f.write(f'Chi-square değeri: {results["chi2"]:.4f}\n')
        f.write(f'Serbestlik derecesi: {results["dof"]}\n')
        f.write(f'p-değeri: {results["p_value"]:.4f}\n')
        if results["p_value"] < 0.05:
            f.write('Yorum: İstatistiksel olarak anlamlı ilişki bulundu (p < 0.05)\n')
        else:
            f.write('Yorum: İstatistiksel olarak anlamlı ilişki bulunamadı (p >= 0.05)\n')
        f.write('\n')

    # Veri dağılımları
    f.write('\nVERİ DAĞILIMLARI\n')
    f.write('================\n\n')
    for var in categorical_vars:
        temp_data = data.copy()
        temp_data[var] = temp_data[var].fillna('Not Available')
        
        if var == 'who_tumor_grade':
            temp_data[var] = temp_data[var].astype(str)
            temp_data[var] = temp_data[var].replace('nan', 'Not Available')
        
        f.write(f'\n{var} Dağılımı (Tüm Veriler):\n')
        f.write('-' * (len(var) + 20) + '\n')
        dist_table_all = pd.crosstab(temp_data['Cluster'], temp_data[var])
        f.write(dist_table_all.to_string())
        f.write('\n\n')
        
        valid_data = temp_data[~temp_data[var].isin(['Not Available', 'Not Performed', 'No alteration'])]
        if len(valid_data) > 0:
            f.write(f'\n{var} Dağılımı (Sadece Geçerli Veriler):\n')
            f.write('-' * (len(var) + 32) + '\n')
            dist_table_valid = pd.crosstab(valid_data['Cluster'], valid_data[var])
            f.write(dist_table_valid.to_string())
            f.write('\n\n')

print("Genetik analiz tamamlandı. Görsel ve istatistiksel sonuçlar genetichisto/ klasörüne kaydedildi.")
print("Her değişken için hem tüm verilerle hem de sadece geçerli verilerle analiz yapıldı.")
print("Detaylı istatistiksel sonuçlar 'genetichisto/cluster_genetic_analysis_results.txt' dosyasına kaydedildi.")
