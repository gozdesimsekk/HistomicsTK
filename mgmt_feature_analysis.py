import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import StandardScaler

# Verileri yükle
clinical_data = pd.read_csv('merged_clinical_radiomic_data.csv')
cluster_data = pd.read_csv('cluster_results_selected_features.csv')

# Verileri birleştir
analysis_df = pd.merge(clinical_data, cluster_data, on='ImageID')

# Klinik özellikleri seç
clinical_features = ['Age_at_scan_years', 'Gender', 'Survival_from_surgery_days_UPDATED', 
                    'Survival_Status', 'IDH1', 'MGMT', 'KPS', 'GTR_over90percent']

# 1. Her küme için klinik özelliklerin özeti
print("Kümelerin Klinik Özellikleri:")
for feature in clinical_features:
    if analysis_df[feature].dtype in ['int64', 'float64']:
        # Sayısal özellikler için ortalama ve std
        stats = analysis_df.groupby('Cluster')[feature].agg(['mean', 'std']).round(2)
        print(f"\n{feature} özeti:")
        print(stats)
    else:
        # Kategorik özellikler için frekans
        for cluster in analysis_df['Cluster'].unique():
            cluster_data = analysis_df[analysis_df['Cluster'] == cluster]
            print(f"\nCluster {cluster} - {feature} dağılımı:")
            print(cluster_data[feature].value_counts(normalize=True).round(3))

# 2. İstatistiksel testler
print("\nİstatistiksel Analiz Sonuçları:")

# Sayısal özellikler için ANOVA
numerical_features = [f for f in clinical_features 
                     if analysis_df[f].dtype in ['int64', 'float64']]
for feature in numerical_features:
    groups = [group[feature].dropna() for name, group 
             in analysis_df.groupby('Cluster')]
    f_val, p_val = f_oneway(*groups)
    print(f"\n{feature} - ANOVA:")
    print(f"F-value: {f_val:.2f}, p-value: {p_val:.4f}")

# Kategorik özellikler için Ki-kare
categorical_features = [f for f in clinical_features 
                       if analysis_df[f].dtype not in ['int64', 'float64']]
for feature in categorical_features:
    contingency = pd.crosstab(analysis_df['Cluster'], analysis_df[feature])
    chi2, p_val, _, _ = chi2_contingency(contingency)
    print(f"\n{feature} - Chi-square:")
    print(f"Chi2: {chi2:.2f}, p-value: {p_val:.4f}")

# 3. Görselleştirmeler
# Yaş dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_df, x='Cluster', y='Age_at_scan_years')
plt.title('Kümelere Göre Yaş Dağılımı')
plt.savefig('cluster_age_distribution.png')

# Sağkalım analizi
plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_df, x='Cluster', 
            y='Survival_from_surgery_days_UPDATED')
plt.title('Kümelere Göre Sağkalım Dağılımı')
plt.savefig('cluster_survival_distribution.png')

# Kategorik değişkenler için heatmap
categorical_proportions = {}
for feature in categorical_features:
    prop_table = pd.crosstab(analysis_df['Cluster'], 
                            analysis_df[feature], 
                            normalize='index')
    categorical_proportions[feature] = prop_table

plt.figure(figsize=(15, 8))
sns.heatmap(pd.concat(categorical_proportions, axis=1), 
            annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Kümelere Göre Kategorik Değişken Dağılımları')
plt.savefig('cluster_categorical_heatmap.png')

# 4. Özellik önem analizi
print("\nKümeleri En İyi Ayıran Özellikler:")
feature_cols = [col for col in analysis_df.columns 
                if col not in ['ImageID', 'PatientID', 'Cluster']]
feature_importance = []

for feature in feature_cols:
    if analysis_df[feature].dtype in ['int64', 'float64']:
        groups = [group[feature].dropna() for name, group 
                 in analysis_df.groupby('Cluster')]
        try:
            f_val, p_val = f_oneway(*groups)
            feature_importance.append({
                'Feature': feature,
                'F_value': f_val,
                'p_value': p_val
            })
        except:
            continue

feature_importance_df = pd.DataFrame(feature_importance)
significant_features = feature_importance_df[
    feature_importance_df['p_value'] < 0.05
].sort_values('F_value', ascending=False)

print("\nAnlamlı Özellikler (p<0.05):")
print(significant_features[['Feature', 'F_value', 'p_value']].to_string()) 