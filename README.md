Link :https://2005prajwal-sct-ml-02-app-6rjifj.streamlit.app/


# 🛍️ Mall Customer Segmentation (K-means Clustering)

This project applies **K-means clustering** to segment customers of a mall based on their **Annual Income** and **Spending Score**.

## 🚀 Features
- Trains a clustering model using KMeans
- Segments customers into distinct groups
- Interactive **Streamlit app** to visualize clusters

## 📂 Project Structure
- `Mall_Customers.csv` → Dataset
- `kmeans_clustering.py` → Training script
- `app.py` → Streamlit app
- `requirements.txt` → Dependencies

## ▶️ Run Locally
```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # (Windows)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python kmeans_clustering.py

# 4. Run app
streamlit run app.py

