import streamlit as st
import joblib
import pandas as pd

from utils.recommender import (
    load_test_data,
    load_item_data,
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
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎬 Ground Truth (Data Test)")
        if gt_user.empty:
            st.warning("Pengguna tidak memiliki data pada test set.")
        else:
            st.dataframe(gt_user[['movie_id', 'title', 'rating']])

    with col2:
        st.markdown(f"### 📊 Rekomendasi (Top-{top_n})")
        st.dataframe(recommendations[['movie_id', 'title']])

else:
    st.info("Silakan masukkan parameter dan klik tombol untuk melihat hasil rekomendasi.")
