import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv('all_density_mean_features.csv')

# ImageID'yi ayır ve feature'ları seç
features = df.drop('ImageID', axis=1)
image_ids = df['ImageID'].values

# Verileri normalize et
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# K-means ile etiketler oluştur (feature selection için)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(normalized_features)

# 1. ANOVA F-değeri ile özellik seçimi
k_best_features = 20  # 10'dan 20'ye çıkarıldı
selector_f = SelectKBest(score_func=f_classif, k=k_best_features)
selector_f.fit(normalized_features, labels)
f_scores = pd.DataFrame({
    'Feature': features.columns,
    'F_Score': selector_f.scores_
})
f_scores = f_scores.sort_values('F_Score', ascending=False)

# 2. Mutual Information ile özellik seçimi
selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_best_features)
selector_mi.fit(normalized_features, labels)
mi_scores = pd.DataFrame({
    'Feature': features.columns,
    'MI_Score': selector_mi.scores_
})
mi_scores = mi_scores.sort_values('MI_Score', ascending=False)

# 3. Random Forest ile özellik önem skorları
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(normalized_features, labels)
importance_scores = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf.feature_importances_
})
importance_scores = importance_scores.sort_values('Importance', ascending=False)

# Görselleştirme
plt.figure(figsize=(15, 15))  # Boyutu artırdık

# 1. F-scores
plt.subplot(3, 1, 1)
sns.barplot(x='F_Score', y='Feature', data=f_scores.head(20))  # 20 özellik göster
plt.title('Top 20 Features (ANOVA F-score)', fontsize=12)

# 2. Mutual Information scores
plt.subplot(3, 1, 2)
sns.barplot(x='MI_Score', y='Feature', data=mi_scores.head(20))  # 20 özellik göster
plt.title('Top 20 Features (Mutual Information)', fontsize=12)

# 3. Random Forest importance
plt.subplot(3, 1, 3)
sns.barplot(x='Importance', y='Feature', data=importance_scores.head(20))  # 20 özellik göster
plt.title('Top 20 Features (Random Forest Importance)', fontsize=12)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

# En önemli özellikleri seç (her yöntemden top 20)
top_features_f = set(f_scores.head(20)['Feature'])
top_features_mi = set(mi_scores.head(20)['Feature'])
top_features_rf = set(importance_scores.head(20)['Feature'])

# Birleştir ve tekrar eden özellikleri kaldır
selected_features = list(top_features_f.union(top_features_mi, top_features_rf))

# Seçilen özelliklerle yeni bir DataFrame oluştur
selected_df = df[['ImageID'] + selected_features]
selected_df.to_csv('selected_features.csv', index=False)

# Sonuçları yazdır
print("\nSeçilen özellik sayıları:")
print(f"ANOVA F-test: {len(top_features_f)} özellik")
print(f"Mutual Information: {len(top_features_mi)} özellik")
print(f"Random Forest: {len(top_features_rf)} özellik")

# Özellik örtüşmelerini hesapla
f_mi_overlap = len(top_features_f.intersection(top_features_mi))
f_rf_overlap = len(top_features_f.intersection(top_features_rf))
mi_rf_overlap = len(top_features_mi.intersection(top_features_rf))
all_overlap = len(top_features_f.intersection(top_features_mi, top_features_rf))

print("\nÖzellik örtüşmeleri:")
print(f"ANOVA ve MI arasında: {f_mi_overlap} özellik")
print(f"ANOVA ve RF arasında: {f_rf_overlap} özellik")
print(f"MI ve RF arasında: {mi_rf_overlap} özellik")
print(f"Üç yöntem arasında: {all_overlap} özellik")

print(f"\nToplam benzersiz özellik sayısı: {len(selected_features)}")
print("\nSeçilen özellikler:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

print("\nSeçilen özellikler 'selected_features.csv' dosyasına kaydedildi.")
print("Özellik önem grafikleri 'feature_importance.png' dosyasına kaydedildi.") 