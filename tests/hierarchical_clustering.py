import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap

# Veriyi oku
df = pd.read_csv('all_density_mean_features.csv')

# ImageID'yi index olarak ayarla ve feature'ları seç
features = df.drop('ImageID', axis=1)
image_ids = df['ImageID'].values  # Series yerine numpy array kullan

# Verileri normalize et
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Hierarchical Clustering uygula
# Ward yöntemi genellikle en iyi sonuçları verir
linkage_matrix = linkage(normalized_features, method='ward')

# 2x2 subplot oluştur
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 1. Dendogram çizimi
dendrogram(linkage_matrix, ax=ax1, labels=image_ids, leaf_rotation=90)
ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=14)
ax1.set_xlabel('Sample ID', fontsize=12)
ax1.set_ylabel('Distance', fontsize=12)

# Belirli bir mesafede cluster'ları oluştur
max_dist = 15  # Bu değeri dendograma bakarak ayarlayabiliriz
cluster_labels = fcluster(linkage_matrix, max_dist, criterion='distance')

# PCA ile 2 boyuta indir
pca = PCA(n_components=2)
features_2d_pca = pca.fit_transform(normalized_features)

# 2. PCA görselleştirmesi
scatter2 = ax2.scatter(features_2d_pca[:, 0], features_2d_pca[:, 1], 
                      c=cluster_labels, cmap='viridis', s=100)
ax2.set_title('PCA Visualization of Clusters', fontsize=14)
ax2.set_xlabel('First Principal Component', fontsize=12)
ax2.set_ylabel('Second Principal Component', fontsize=12)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Her nokta için ImageID'yi ekle (PCA)
for i, txt in enumerate(image_ids):
    ax2.annotate(txt, (features_2d_pca[i, 0], features_2d_pca[i, 1]), 
                fontsize=8, xytext=(5, 5), textcoords='offset points')

# UMAP ile 2 boyuta indir
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(normalized_features)

# 3. UMAP görselleştirmesi
scatter3 = ax3.scatter(embedding[:, 0], embedding[:, 1], 
                      c=cluster_labels, cmap='viridis', s=100)
ax3.set_title('UMAP Visualization of Clusters', fontsize=14)
ax3.set_xlabel('UMAP1', fontsize=12)
ax3.set_ylabel('UMAP2', fontsize=12)
plt.colorbar(scatter3, ax=ax3, label='Cluster')

# Her nokta için ImageID'yi ekle (UMAP)
for i, txt in enumerate(image_ids):
    ax3.annotate(txt, (embedding[i, 0], embedding[i, 1]), 
                fontsize=8, xytext=(5, 5), textcoords='offset points')

# 4. Heatmap görselleştirmesi
# Örnekleri cluster'lara göre sırala
sorted_indices = np.argsort(cluster_labels)
sorted_features = normalized_features[sorted_indices]
sorted_ids = image_ids[sorted_indices]

# Heatmap çiz
sns.heatmap(sorted_features, ax=ax4, cmap='viridis', 
            xticklabels=features.columns, yticklabels=sorted_ids)
ax4.set_title('Feature Heatmap (Sorted by Clusters)', fontsize=14)
ax4.set_xlabel('Features', fontsize=12)
ax4.set_ylabel('Samples', fontsize=12)
plt.xticks(rotation=90)

plt.tight_layout()
# Sonuçları kaydet
plt.savefig('hierarchical_clustering_results.png', dpi=300, bbox_inches='tight')

# Cluster bilgilerini DataFrame olarak kaydet ve sırala
results_df = pd.DataFrame({
    'ImageID': image_ids,
    'Cluster': cluster_labels
})
results_df = results_df.sort_values('Cluster')  # Cluster'a göre sırala
results_df.to_csv('hierarchical_cluster_results.csv', index=False)

# # Cluster istatistiklerini yazdır
unique_clusters = np.unique(cluster_labels)
print(f"\nToplam {len(unique_clusters)} cluster bulundu:")
for cluster in sorted(unique_clusters):
    cluster_size = np.sum(cluster_labels == cluster)
    print(f"\nCluster {cluster}: {cluster_size} örnek")
    # Her cluster'daki örnekleri yazdır
    cluster_samples = results_df[results_df['Cluster'] == cluster]['ImageID'].tolist()
    print(f"Örnekler: {', '.join(cluster_samples)}")

# Cluster istatistiklerini kaydet
cluster_statistics = []

for cluster in sorted(unique_clusters):
    cluster_mask = cluster_labels == cluster
    cluster_features = features.iloc[cluster_mask]
    
    for feature in cluster_features.columns:
        # Eksik veri kontrolü ve loglama
        if cluster_features[feature].isnull().any():
            mean_value = cluster_features[feature].mean()
            std_value = np.nan  # Eksik veri olduğu için standart sapma hesaplanamaz
            print(f"Cluster {cluster}: '{feature}' özelliğinde eksik (NaN) değerler var. Ortalama: {mean_value:.2f}, Standart Sapma: NaN")
        else:
            mean_value = cluster_features[feature].mean()
            std_value = cluster_features[feature].std() if cluster_features[feature].nunique() > 1 else 0  # Sabit değerler için 0
            print(f"Cluster {cluster}: '{feature}' özelliği için Ortalama: {mean_value:.2f}, Standart Sapma: {std_value:.2f}")
        
        cluster_statistics.append({
            'Cluster': cluster,
            'Feature': feature,
            'Mean': mean_value,
            'Standard Deviation': std_value
        })
cluster_statistics_df = pd.DataFrame(cluster_statistics)

cluster_statistics_df.to_csv('cluster_statistics.csv', index=False)

print("Cluster istatistikleri 'cluster_statistics.csv' dosyasına kaydedildi.")


print("\nClustering tamamlandı. Sonuçlar 'hierarchical_clustering_results.png' ve 'hierarchical_cluster_results.csv' dosyalarına kaydedildi.") 