import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("Mall Customer Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    # Number of clusters slider
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)

    # Button to run clustering
    if st.button("Run KMeans"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[['Annual Income (k$)','Spending Score (1-100)']])
        st.success("Clustering done!")
        st.write(data.head())
