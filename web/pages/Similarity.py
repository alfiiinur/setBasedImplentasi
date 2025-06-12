import streamlit as st
import joblib
import pandas as pd

from utils.recommender import (
    load_test_data,
    load_item_data,
    load_training_data,
    generate_recommendations,
)

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("🔍 Rekomendasi Berdasarkan Fungsi Similaritas")

# ========================
# Form Input
# ========================
similarity_method = st.selectbox("📌 Pilih Fungsi Similaritas", ["Jaccard", "Relevant Jaccard"])
top_n = st.selectbox("🔢 Top-N Rekomendasi", list(range(1, 21)), index=4)  # 5 default
user_id = st.number_input("👤 Masukkan User ID", min_value=1, max_value=1982, step=1)

# ========================
# Proses Rekomendasi
# ========================
if st.button("🎯 Proses Rekomendasi"):
    # Load data dan model
    test_data = load_test_data()  # dari file: data/u.test
    training_data = load_training_data() #dari file data/u.base
    item_data = load_item_data()  # dari file: data/u.item

    model_path = "utils/model/JacModel.joblib" if similarity_method == "Jaccard" else "utils/model/RJModel.joblib"
    model = joblib.load(model_path)




    # Get rekomendasi dan nilai NDCG dari model
    try:
        ndcg_value = model.get("ndcg", {}).get(top_n, 0.0)
    except Exception:
        ndcg_value = 0.0

    recommendations = generate_recommendations(user_id, top_n, similarity_method)


    # Ground truth user
    gt_user = test_data[test_data['user_id'] == user_id]
    gt_user = gt_user.merge(item_data, on="movie_id", how="left")

    # data training
    gt_user_training = training_data[training_data['user_id'] == user_id]
    gt_user_training = gt_user_training.merge(item_data, on="movie_id",how="left")

    # intersection

    # Cari irisan antara rekomendasi dan ground truth
    intersection = pd.merge(recommendations, gt_user[['movie_id']], on="movie_id", how="inner")
    intersection = intersection.merge(item_data, on="movie_id", how="left")


    # ========================
    # Informasi Ringkasan
    # ========================
    st.markdown("### 📝 **Informasi Pengguna & Parameter**")
    st.markdown(f"- **Metode Similarity:** {similarity_method}")
    st.markdown(f"- **Top-N Rekomendasi:** {top_n}")
    st.markdown(f"- **User ID:** {user_id}")
    st.markdown(f"- **Nilai NDCG:** {ndcg_value:.4f}")

    # ========================
    # Tampilkan Data
    # ========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Ground Truth")
        st.divider()
        st.text(f"Data rekomendasi yang cocok dengan data test (interseksi) sebanyak {len(intersection)} item.")
        if gt_user.empty:
            st.warning("Pengguna tidak memiliki data pada test set.")
        else:
            st.dataframe(gt_user[['movie_id', 'title', 'rating']])

    with col2:
        st.markdown("###  Data Training")
        st.divider()
        st.text(f"Data training yaitu data item yang telah diberikan rating oleh target user yang didapatkan dari user -  {user_id} ")
        if gt_user_training.empty:
            st.warning("Pengguna tidak memiliki data training")
        else:
            st.dataframe(gt_user_training[['movie_id', 'title', 'rating']])

    with col3: 
        st.markdown(f"### Data Rekomendasi")
        st.divider()
        st.text("Data test dari target user yang didapatkan dari dataset dan untuk digunakan di dalam evaluasi hasil rekomendasi metode. Data Test didapatkan  sebanyak ")
        if intersection.empty:
            st.warning("Pengguna tidak memiliki rekomendasi")
        else:
            st.dataframe(intersection[['movie_id', 'title']])
       
else:
    st.info("Silakan masukkan parameter dan klik tombol untuk melihat hasil rekomendasi.")
