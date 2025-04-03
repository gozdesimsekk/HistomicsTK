import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# histomic feature set with an ImageID
df = pd.read_csv('all_density_mean_features.csv')
df = df.drop(columns=['ImageID'])

# Calculating the mean, standard deviation, variance, and Coefficient of Variation (CV)
results = []
for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    variance = df[col].var()
    if mean != 0:  
        cv = std / mean  
    else:
        cv = None
    results.append({
        'Feature': col,
        'Mean': mean,
        'Std': std,
        'Variance': variance,
        'CV': cv
    })

# RESULTS
results_df = pd.DataFrame(results)

# Sorting
sorted_results_df = results_df.sort_values(by='CV', ascending=False)

# print("Features ranked by Coefficient of Variation (CV):")
# print(sorted_results_df[['Feature', 'Mean', 'Std', 'Variance', 'CV']])

#VISUALIZE
sns.set(style="whitegrid") 
plt.figure(figsize=(14, 8))

bars = plt.bar(sorted_results_df['Feature'], sorted_results_df['CV'], 
               color=sns.color_palette("Blues_r", len(sorted_results_df)), edgecolor='black')

for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.09, round(yval, 3), 
             ha='center', va='bottom', fontsize=10, fontweight='600', color='black', rotation=90)

plt.xticks(rotation=90, ha='center', fontsize=12, fontweight= '600')  
plt.xlabel("Features", fontsize=14, fontweight= '600')
plt.ylabel("CV", fontsize=14, fontweight= '600')
plt.title("Coefficient of Variation (CV) Values for All Features", fontsize=16, fontweight= '600')
plt.subplots_adjust(bottom=0.2)

# Save PNG
plt.tight_layout() 
plt.savefig('coefficient_of_variation.png', format='png', dpi=300)  
plt.show()
