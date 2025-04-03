import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def analyze_clusters(features_file, cluster_results_file):
    """
    Kümelerin detaylı analizini gerçekleştirir.
    """
    # Verileri yükle
    features_df = pd.read_csv(features_file)
    cluster_results = pd.read_csv(cluster_results_file)
    
    # ImageID'yi index olarak ayarla
    features = features_df.set_index('ImageID')
    
    # Verileri normalize et
    scaler = StandardScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Küme etiketlerini ekle
    normalized_features['Cluster'] = cluster_results.set_index('ImageID')['Cluster']
    
    # 1. Küme Karakteristik Analizi
    cluster_means = normalized_features.groupby('Cluster').mean()
    cluster_stds = normalized_features.groupby('Cluster').std()
    
    # 2. Öznitelik Önem Analizi
    feature_importance = calculate_feature_importance(normalized_features)
    
    # 3. Görselleştirmeler
    create_visualizations(normalized_features, cluster_means, feature_importance)
    
    # 4. Detaylı rapor oluştur
    generate_report(cluster_means, cluster_stds, feature_importance)

def calculate_feature_importance(data):
    """
    Her özniteliğin kümeler arası ayırt ediciliğini hesaplar
    """
    feature_importance = pd.DataFrame(
        index=data.columns[:-1],  # 'Cluster' sütununu hariç tut
        columns=['F_value', 'importance_score']
    )
    
    for feature in data.columns[:-1]:
        # Kümeler arası F-testi
        groups = [group[feature].values for name, group in data.groupby('Cluster')]
        f_value = f_oneway(*groups)[0]
        
        # Öznitelik önem skoru
        importance = np.abs(data.groupby('Cluster')[feature].mean().std())
        
        feature_importance.loc[feature] = [f_value, importance]
    
    return feature_importance.sort_values('F_value', ascending=False)

def create_visualizations(data, cluster_means, feature_importance):
    """
    Analiz sonuçlarını görselleştirir
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Küme Profilleri Heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(cluster_means, cmap='viridis', center=0)
    plt.title('Küme Profilleri')
    
    # 2. En Önemli Öznitelikler
    plt.subplot(2, 1, 2)
    top_features = feature_importance.head(10)
    sns.barplot(x=top_features.index, y='F_value', data=top_features)
    plt.xticks(rotation=45)
    plt.title('En Ayırt Edici 10 Öznitelik')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis_results.png')

def generate_report(cluster_means, cluster_stds, feature_importance):
    """
    Detaylı analiz raporu oluşturur
    """
    report = []
    report.append("GBM Alt Grup Karakterizasyon Raporu\n")
    
    # Her küme için karakteristik özellikleri belirle
    for cluster in cluster_means.index:
        report.append(f"\nKüme {cluster} Karakteristikleri:")
        
        # En belirgin özellikler (z-score > 1)
        significant_features = cluster_means.loc[cluster][
            np.abs(cluster_means.loc[cluster]) > 1
        ].sort_values(ascending=False)
        
        report.append("Belirgin Özellikler:")
        for feature, value in significant_features.items():
            std = cluster_stds.loc[cluster, feature]
            report.append(f"- {feature}: {value:.2f} (±{std:.2f})")
    
    # En ayırt edici özellikler
    report.append("\nEn Ayırt Edici Özellikler:")
    top_features = feature_importance.head(10)
    for feature in top_features.index:
        report.append(f"- {feature}: F={top_features.loc[feature, 'F_value']:.2f}")
    
    # Raporu dosyaya kaydet
    with open('cluster_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    # Dosya yollarını belirt
    features_file = 'selected_features.csv'
    cluster_results_file = 'cluster_results_selected_features.csv'
    
    # Analizi gerçekleştir
    analyze_clusters(features_file, cluster_results_file)
    
    print("Analiz tamamlandı. Sonuçlar 'cluster_analysis_results.png' ve 'cluster_analysis_report.txt' dosyalarına kaydedildi.")

if __name__ == "__main__":
    main() 