import streamlit as st 
import pickle 
import numpy as np

# تحميل الموديلات
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
dbscan_model = pickle.load(open("dbscan_model.pkl", "rb"))

st.title("🤖 Prediction & Clustering App")

# اختيار نوع الموديل
model_choice = st.selectbox(
    "اختر نوع الموديل:",
    ["KMeans (Clustering)", "DBSCAN (Clustering)"]
)

annual_income = st.number_input("Enter Annual Income ($):", min_value=15.0, max_value=137.0, step=1.0)
spending_score = st.number_input("Enter Spending Score (1-100):", min_value=1.0, max_value=100.0, step=1.0)

input_features = np.array([[annual_income, spending_score]])

# لو اختار KMeans
if model_choice == "KMeans (Clustering)":
    if st.button("Predict Cluster (KMeans)"):
        cluster = kmeans_model.predict(input_features)[0]+1
        st.success(f"Customer belongs to Cluster: {cluster}")

# لو اختار DBSCAN
elif model_choice == "DBSCAN (Clustering)":
    if st.button("Predict Cluster (DBSCAN)"):
        # DBSCAN ملهاش predict --> هنعمل fit على البيانات الجديدة
        cluster = dbscan_model.fit_predict(input_features)[0]
        st.success(f"Customer belongs to Cluster: {cluster}")
