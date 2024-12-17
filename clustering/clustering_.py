# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
"num-of-doors", "body-style", "drive-wheels", "engine-location",
"wheel-base", "length", "width", "height", "curb-weight",
"engine-type", "num-of-cylinders", "engine-size", "fuel-system",
"bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
"city-mpg", "highway-mpg", "price"]
df = pd.read_csv('/content/imports-85.data', header=None, names=headers)
df

# %%
df.replace('?', np.nan, inplace=True)

# %%
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
plt.scatter(x=df['price'], y=df['horsepower'])

# %%
X = df[['price', 'horsepower']].astype(float).values


# %%
from sklearn.cluster import KMeans

# %%
km3 = KMeans(n_clusters=3, random_state=42)
km3.fit(X)

# %%
km3.transform(X)

# %%
km3.labels_

# %%
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)

# %%
km3_scaled = KMeans(n_clusters=3, random_state=42)
km3_scaled.fit(X_scaled)

# %%
km3_scaled.transform(X_scaled)

# %%
km3_scaled.labels_

# %%
plt.scatter(X[:, 0], X[:, 1], c=km3.labels_)
plt.scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1], marker = "X", color="red");

# %%
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km3_scaled.labels_)
plt.scatter(km3_scaled.cluster_centers_[:, 0], km3_scaled.cluster_centers_[:, 1], marker = "X", color="red");

# %% [markdown]
# ## Elbow

# %%
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# %%
inertias = []
for k in range(2, 11):
  km = KMeans(n_clusters=k, n_init=20, random_state=42)
  km.fit(X_scaled)
  inertias.append(km.inertia_)


# %%
inertias

# %%
plt.plot(range(2, 11), inertias)

# %% [markdown]
# ## Silhouette

# %%
s_scores = []
for k in range(2, 11):
  km = KMeans(n_clusters=k, n_init=20, random_state=42)
  km.fit(X_scaled)
  score = silhouette_score(X_scaled, km.labels_)
  s_scores.append(score)

# %%
s_scores

# %%
plt.plot(range(2, 11), s_scores)

# %% [markdown]
# ## **Calinski**

# %%
c_scores = []
for k in range(2, 11):
  km = KMeans(n_clusters=k, n_init=20, random_state=42)
  km.fit(X_scaled)
  score = calinski_harabasz_score(X_scaled, km.labels_)
  c_scores.append(score)

# %%
c_scores

# %%
plt.plot(range(2, 11), c_scores)

# %% [markdown]
# ## 7 Clusters

# %%
km7_scaled = KMeans(n_clusters=7, random_state=42)
km7_scaled.fit(X_scaled)

# %%
km7_scaled.transform(X_scaled)

# %%
km7_scaled.labels_

# %%
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km7_scaled.labels_)
plt.scatter(km7_scaled.cluster_centers_[:, 0], km7_scaled.cluster_centers_[:, 1], marker = "X", color="red");

# %% [markdown]
# ## Boundaries

# %%
from scipy.spatial import Voronoi, voronoi_plot_2d

# %%
vor = Voronoi(km3_scaled.cluster_centers_)
fig = voronoi_plot_2d(vor)

# %% [markdown]
# ## DBSCAN

# %%
from sklearn.cluster import DBSCAN

# %%
X


# %%
X_scaled

# %%
dbs = DBSCAN(eps=0.17, min_samples=5)
dbs.fit(X_scaled)

# %%
dbs.labels_

# %%
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbs.labels_)

# %% [markdown]
# ## Nearest Neighbors

# %%
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %%
mlb = pd.read_csv('/content/mlb_batting_cleaned.csv')

# %%
mlb

# %%
def find_closest_players(df):
  player_name = input("Enter a player's name: ")
  player_names = df['Name']
  features = df.drop(columns=['Name'])


  num_cols = features.select_dtypes(exclude='object')
  categorical_cols = features.select_dtypes(include='object').columns

  ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

  encoded = ohe.fit_transform(df[categorical_cols])
  encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols))

  mms = MinMaxScaler()
  features_scaled = mms.fit_transform(num_cols)

  nn = NearestNeighbors(n_neighbors=3)
  nn.fit(features_scaled)

  player_index = player_names[player_names == player_name].index[0]

  dist, neighbors = nn.kneighbors([features_scaled[player_index]])

  closest_neighbors = neighbors[0][1:]
  closest_players = player_names.iloc[closest_neighbors].values

  print("The first closest player: ", closest_players[0])
  print("The second closest player: ", closest_players[1])



# %%
find_closest_players(mlb)

# %%



