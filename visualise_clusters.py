import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load embeddings and texts from npz file
loaded = np.load('tweet_embeddings.npz', allow_pickle=True)
embeddings = loaded['embeddings']
texts = loaded['texts']

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Step 2: Perform clustering (here using KMeans, you can replace with other algorithms)
n_clusters = int(input("Enter the number of clusters: "))  # You can change the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scaled_embeddings)
labels = kmeans.labels_

# Step 3: Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(scaled_embeddings)

# Step 4: Create Plotly visualization
df = {
    'x': reduced_embeddings[:, 0],
    'y': reduced_embeddings[:, 1],
    'labels': labels,
    'texts': texts
}

fig = px.scatter(df, x='x', y='y', color='labels', hover_data=['texts'])
fig.show()
