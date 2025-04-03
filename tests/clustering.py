import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import leidenalg
import igraph as ig
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap

# Veriyi oku
df = pd.read_csv('all_density_mean_features.csv')

# ImageID'yi index olarak ayarla ve feature'ları seç
features = df.drop('ImageID', axis=1)
image_ids = df['ImageID']

# Verileri normalize et
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Öklid mesafelerini hesapla
distances = euclidean_distances(normalized_features)

# Mesafeleri benzerliğe çevir (daha küçük mesafe = daha yüksek benzerlik)
similarities = np.exp(-distances / distances.std())

# Benzerlik matrisinin simetrik olduğundan emin ol
similarities = (similarities + similarities.T) / 2

# Graph oluştur
G = ig.Graph.Weighted_Adjacency(similarities.tolist(), mode="undirected", attr="weight", loops=False)

# Leiden algoritmasını uygula (resolution parametresini düşür)
partition = leidenalg.find_partition(G, 
                                   leidenalg.RBConfigurationVertexPartition,
                                   weights=G.es['weight'],
                                   resolution_parameter=1.0)  # Cluster sayısını azalttık

# Cluster etiketlerini al
cluster_labels = partition.membership

# UMAP ile 2 boyuta indir
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(normalized_features)

# Görselleştirme için subplot'lar oluştur
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# PCA görselleştirmesi
pca = PCA(n_components=2)
features_2d_pca = pca.fit_transform(normalized_features)

scatter1 = ax1.scatter(features_2d_pca[:, 0], features_2d_pca[:, 1], 
                      c=cluster_labels, cmap='viridis', s=100)
ax1.set_title('PCA Visualization of Clusters', fontsize=14)
ax1.set_xlabel('First Principal Component', fontsize=12)
ax1.set_ylabel('Second Principal Component', fontsize=12)
plt.colorbar(scatter1, ax=ax1, label='Cluster')

# Her nokta için ImageID'yi ekle (PCA)
for i, txt in enumerate(image_ids):
    ax1.annotate(txt, (features_2d_pca[i, 0], features_2d_pca[i, 1]), 
                fontsize=8, xytext=(5, 5), textcoords='offset points')

# UMAP görselleştirmesi
scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                      c=cluster_labels, cmap='viridis', s=100)
ax2.set_title('UMAP Visualization of Clusters', fontsize=14)
ax2.set_xlabel('UMAP1', fontsize=12)
ax2.set_ylabel('UMAP2', fontsize=12)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Her nokta için ImageID'yi ekle (UMAP)
for i, txt in enumerate(image_ids):
    ax2.annotate(txt, (embedding[i, 0], embedding[i, 1]), 
                fontsize=8, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
# Sonuçları kaydet
plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')

# Cluster bilgilerini DataFrame olarak kaydet ve sırala
results_df = pd.DataFrame({
    'ImageID': image_ids,
    'Cluster': cluster_labels
})
results_df = results_df.sort_values('Cluster')  # Cluster'a göre sırala
results_df.to_csv('leiden_cluster_results.csv', index=False)

# Cluster istatistiklerini yazdır
unique_clusters = np.unique(cluster_labels)
print(f"\nToplam {len(unique_clusters)} cluster bulundu:")
for cluster in sorted(unique_clusters):
    cluster_size = np.sum(cluster_labels == cluster)
    print(f"Cluster {cluster}: {cluster_size} örnek")
    # Her cluster'daki örnekleri yazdır
    cluster_samples = results_df[results_df['Cluster'] == cluster]['ImageID'].tolist()
    print(f"Örnekler: {', '.join(cluster_samples)}")
    print()

print("\nClustering tamamlandı. Sonuçlar 'clustering_visualization.png' ve 'leiden_cluster_results.csv' dosyalarına kaydedildi.") 