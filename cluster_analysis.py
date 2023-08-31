import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load tweet embeddings from the npz file
data = np.load('tweet_embeddings.npz')
tweet_embeddings = data['embeddings']

# Elbow Method
inertia_values = []
k_range = range(2, 31)  # You can modify the range as needed

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(tweet_embeddings)
    inertia_values.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure()
plt.plot(k_range, inertia_values, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Silhouette Analysis
silhouette_values = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(tweet_embeddings)
    silhouette_avg = silhouette_score(tweet_embeddings, cluster_labels)
    silhouette_values.append(silhouette_avg)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

# Plotting the Silhouette Analysis graph
plt.figure()
plt.plot(k_range, silhouette_values, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis For Optimal k')
plt.show()
