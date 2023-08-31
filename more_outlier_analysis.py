from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Load your tweet embeddings and texts
data = np.load('tweet_embeddings.npz')
embeddings = data['embeddings']
texts = data['texts']

# Function to get outliers using Isolation Forest
def isolation_forest_outliers(embeddings, contamination=0.05):
    clf = IsolationForest(contamination=contamination)
    clf.fit(embeddings)
    isof_outliers = clf.predict(embeddings)
    return np.where(isof_outliers == -1)[0]

# Function to get outliers using Local Outlier Factor
def lof_outliers(embeddings, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof_outliers = lof.fit_predict(embeddings)
    return np.where(lof_outliers == -1)[0]

# Get user input for parameters
contamination = float(input("Enter the contamination parameter for Isolation Forest (default 0.05): "))
n_neighbors = int(input("Enter the number of neighbors for LOF (default 20): "))

# Get outliers
isof_indices = isolation_forest_outliers(embeddings, contamination)
lof_indices = lof_outliers(embeddings, n_neighbors)

# Print the indices of the outliers
print(f"Outliers detected by Isolation Forest at indices: {isof_indices}")
print(f"Outliers detected by Local Outlier Factor at indices: {lof_indices}")

# You can then use these indices to retrieve the corresponding tweets for further analysis.
# Print the tweets at the outlier indices
for idx in isof_indices[:10]:
    print(f"ISOF Outlier Index: {idx}")
    tweet_text = texts[idx]
    print(f"Tweet: {tweet_text}...")

print("\n\n--------------------------------\n\n")

# Print the tweets at the outlier indices
for idx in lof_indices[:10]:
    print(f"LOF Outlier Index: {idx}")
    tweet_text = texts[idx]
    print(f"Tweet: {tweet_text}...")
