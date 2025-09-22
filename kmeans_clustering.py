import pandas as pd
from sklearn.cluster import KMeans

# Load your dataset
data = pd.read_csv("mall_customers.csv")

# Ask user how many clusters
n_clusters = int(input("Enter the number of clusters: "))

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Annual Income (k$)','Spending Score (1-100)']])

print("Clustering done! Here are the first 5 rows with cluster labels:")
print(data.head())

