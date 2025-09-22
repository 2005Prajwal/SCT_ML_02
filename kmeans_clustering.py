import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Save model & clustered data
joblib.dump(kmeans, "kmeans_model.pkl")
df.to_csv("clustered_customers.csv", index=False)

print("✅ Model trained and saved as kmeans_model.pkl")
