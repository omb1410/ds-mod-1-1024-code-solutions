# %%
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('mnist.csv')

# %%
df

# %%
from sklearn.decomposition import PCA

# %%
pca = PCA(n_components=0.9)
pca.fit(df)
df_pca = pca.transform(df)

# %%
df

# %%
df_pca.shape

# %%
df_pca = pd.DataFrame(df_pca)

# %%
df_pca

# %%
pca_explained = pca.explained_variance_
pca_explained

# %%
plt.plot(pca_explained)

# %%
pca.explained_variance_ratio_

# %%
pca.inverse_transform(df_pca)

# %%



