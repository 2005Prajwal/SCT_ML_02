import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model & dataset
model = joblib.load("kmeans_model.pkl")
df = pd.read_csv("clustered_customers.csv")

st.title("🛍️ Mall Customer Segmentation")
st.write("K-means clustering based customer segmentation app")

# Show dataset preview
if st.checkbox("Show dataset"):
    st.dataframe(df.head())

# Visualize clusters
st.subheader("Customer Segmentation")
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="rainbow"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
st.pyplot(plt)
