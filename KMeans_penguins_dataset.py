# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("data/penguins.csv")

# EDA -- preview data, check data types
penguins_df.head()
penguins_df.shape
penguins_df.info()

# EDA -- check missing values
penguins_df.isna().sum().sort_values()
threshold = len(penguins_df) * 0.5
cols_to_drop = penguins_df.columns[penguins_df.isna().sum() <= threshold]
penguins_df.dropna(subset = cols_to_drop, inplace = True)

# check missing values after dropping them.
penguins_df.isna().sum().sort_values()

# EDA -- check and address outliers
#sns.boxplot(penguins_df)
q3 = np.percentile(penguins_df['flipper_length_mm'], 75)
q1 = np.percentile(penguins_df['flipper_length_mm'], 25)
iqr = q3 - q1
outlier_threshold = 1.5
lower = q1 - (outlier_threshold * iqr)
upper = q3 + (outlier_threshold * iqr)
penguins_clean = penguins_df[(penguins_df['flipper_length_mm'] >= lower) & (penguins_df['flipper_length_mm'] <= upper)]
sns.boxplot(penguins_clean)

# EDA -- check and address outliers
#sns.boxplot(penguins_df)
q3 = np.percentile(penguins_df['flipper_length_mm'], 75)
q1 = np.percentile(penguins_df['flipper_length_mm'], 25)
iqr = q3 - q1
outlier_threshold = 1.5
lower = q1 - (outlier_threshold * iqr)
upper = q3 + (outlier_threshold * iqr)
penguins_clean = penguins_df[(penguins_df['flipper_length_mm'] >= lower) & (penguins_df['flipper_length_mm'] <= upper)]
sns.boxplot(penguins_clean)

# scaling 
scaling = StandardScaler()
penguins_preprocessed = scaling.fit_transform(penguins_clean)
sns.boxplot(penguins_preprocessed)

# PCA
# find the intrinsic dimension
pca = PCA()
pca.fit(penguins_preprocessed)
ratio = pca.explained_variance_ratio_
print(ratio)
pca_threshold = 0.1
n_components = sum(ratio > pca_threshold)
print(n_components)

# performs PCA
pca = PCA(n_components = n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

# find the n_cluster : elbow analysis
range_n_cluster = range(1,10)
inertias = []
for each in range_n_cluster:
  model = KMeans(n_clusters = each, random_state = 42)
  model.fit(penguins_PCA)
  inertias.append(model.inertia_)

# elbow analysis
plt.plot(range_n_cluster,inertias,'-o')
plt.xlabel('number of cluster')
plt.ylabel('inertia')
plt.xticks(range_n_cluster)
plt.plot()

n_cluster = 4

# perform KMeans
kmeans = KMeans(n_clusters = n_cluster, random_state = 42)
kmeans.fit(penguins_PCA)

plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c = kmeans.labels_)
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.show()

penguins_clean['label'] = kmeans.labels_
print(penguins_clean)
num_col = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','label']
stat_penguins = penguins_clean[num_col].groupby('label').mean()
print(stat_penguins)
