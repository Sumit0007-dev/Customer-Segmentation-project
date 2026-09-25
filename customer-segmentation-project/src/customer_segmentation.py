"""
Customer Segmentation using K-Means Clustering
------------------------------------------------
Goal: Group mall customers into meaningful segments based on their
Annual Income and Spending Score, using unsupervised learning
(no labels required).

Pipeline:
1. Load & explore the dataset
2. Select and scale features
3. Find the optimal number of clusters (Elbow Method)
4. Fit K-Means and assign cluster labels
5. Visualize and interpret the segments
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Paths (relative to project root, so this works run from anywhere)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load & explore the dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("First 5 rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# 2. Select & scale features
# -----------------------------
# We cluster on Annual Income and Spending Score, since these two
# variables define customer purchasing behaviour most directly.
features = df[["Annual_Income_k", "Spending_Score"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------------
# 3. Elbow Method to choose k
# -----------------------------
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.xticks(list(k_range))
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved elbow_plot.png -> used to visually pick k (elbow ~ k=5 here)")

# -----------------------------
# 4. Fit K-Means with chosen k
# -----------------------------
OPTIMAL_K = 5  # chosen from the elbow plot

kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nCluster sizes:\n", df["Cluster"].value_counts().sort_index())

# -----------------------------
# 5. Visualize the segments
# -----------------------------
plt.figure(figsize=(8, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

for cluster_id in range(OPTIMAL_K):
    cluster_data = df[df["Cluster"] == cluster_id]
    plt.scatter(
        cluster_data["Annual_Income_k"],
        cluster_data["Spending_Score"],
        s=60,
        c=colors[cluster_id],
        label=f"Cluster {cluster_id}",
        alpha=0.8,
        edgecolors="white",
    )

# plot centroids back in original scale
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    s=250,
    c="black",
    marker="X",
    label="Centroids",
)

plt.title("Customer Segments by Income & Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "customer_segments.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved customer_segments.png -> final cluster visualization")

# -----------------------------
# 6. Interpret each cluster
# -----------------------------
summary = df.groupby("Cluster")[["Age", "Annual_Income_k", "Spending_Score"]].mean().round(1)
summary["Count"] = df["Cluster"].value_counts().sort_index()
print("\nCluster profile summary:\n", summary)

summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))
df.to_csv(os.path.join(OUTPUT_DIR, "mall_customers_with_clusters.csv"), index=False)
print(f"\nSaved cluster_summary.csv and mall_customers_with_clusters.csv to {OUTPUT_DIR}/")
