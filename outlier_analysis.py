from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import scipy.stats as stats


def load_embeddings(path):
    with np.load(path) as data:
        embeddings = data['embeddings']
        texts = data['texts']
    return embeddings, texts


def z_score_outliers(embeddings, texts, threshold=2):
    z_scores = np.abs(stats.zscore(embeddings))
    outliers = np.where(np.any(z_scores > threshold, axis=1))
    return texts[outliers]


def isolation_forest_outliers(embeddings, texts):
    clf = IsolationForest(contamination='auto')
    preds = clf.fit_predict(embeddings)
    outliers = np.where(preds == -1)
    return texts[outliers]


def dbscan_outliers(embeddings, texts):
    dbscan = DBSCAN()
    dbscan.fit(embeddings)
    outliers = np.where(dbscan.labels_ == -1)
    return texts[outliers]


def lof_outliers(embeddings, texts):
    lof = LocalOutlierFactor()
    preds = lof.fit_predict(embeddings)
    outliers = np.where(preds == -1)
    return texts[outliers]


if __name__ == '__main__':
    path = input("Enter the path to your NPZ file: ")
    embeddings, texts = load_embeddings(path)

    print("Performing Z-Score Analysis...")
    z_outliers = z_score_outliers(embeddings, texts)
    print(f"Z-Score Outliers: {z_outliers[:10]}")

    print("Performing Isolation Forest Analysis...")
    iso_outliers = isolation_forest_outliers(embeddings, texts)
    print(f"Isolation Forest Outliers: {iso_outliers[:10]}")

    print("Performing DBSCAN Analysis...")
    dbscan_outliers = dbscan_outliers(embeddings, texts)
    print(f"DBSCAN Outliers: {dbscan_outliers[:10]}")

    print("Performing Local Outlier Factor Analysis...")
    lof_outliers = lof_outliers(embeddings, texts)
    print(f"Local Outlier Factor Outliers: {lof_outliers[:10]}")
