import streamlit as st

from utils.recommender import (
    load_test_data,
    get_mock_predictions,
    get_mock_recommendations
)

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("🔍 Rekomendasi Berdasarkan Fungsi Similaritas")

# ========================
# Form Input
# ========================
pilihFold = st.selectbox("📂 Pilih Fold Data Uji", [1, 2, 3, 4, 5], index=0)
similarity_method = st.selectbox("📌 Pilih Fungsi SImilaritas", ["Jaccard", "Relevant Jaccard"])
top_n = st.selectbox("🔢 Top-N Rekomendasi", [5, 10, 15, 20], index=1)
user_id = st.number_input("👤 Masukkan User ID", min_value=1, max_value=1982, step=1)

# ========================
# Proses Rekomendasi
# ========================
if st.button("🎯 Proses Rekomendasi"):
    # Dummy calls (ganti nanti dengan fungsi aktual)
    ground_truth = load_test_data(pilihFold)
    predictions = get_mock_predictions(user_id)
    recommendations = get_mock_recommendations(user_id, top_n)

    # Filter hanya ground truth milik user terkait
    gt_user = ground_truth[ground_truth['user_id'] == user_id]

    # NDCG dummy (ganti nanti dengan fungsi evaluasi aktual)
    ndcg_value = 0.85  # placeholder

    # ========================
    # Informasi Ringkasan
    # ========================
    st.markdown("### 📝 **Informasi Pengguna & Parameter**")
    st.markdown(f"- **Fold yang dipilih:** {pilihFold}")
    st.markdown(f"- **Metode Similarity:** {similarity_method}")
    st.markdown(f"- **Top-N Rekomendasi:** {top_n}")
    st.markdown(f"- **User ID:** {user_id}")
    st.markdown(f"- **Nilai NDCG:** {ndcg_value:.2f}")

    # ========================
    # Tampilkan Data
    # ========================
    st.markdown("## ✅ Hasil Rekomendasi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎬 Ground Truth (Data Test)")
        if gt_user.empty:
            st.warning("Pengguna tidak memiliki data pada test set yang dipilih.")
        else:
            st.dataframe(gt_user[['movie_id', 'rating']])  # atau tambahkan 'title' jika sudah merge

    with col2:
        st.markdown(f"### 📊 Rekomendasi (Top-{top_n})")
        st.dataframe(recommendations[['movie_id', 'title']])

    # (Opsional) Tampilkan Prediksi Training
    st.markdown("---")
    st.markdown("### 📘 Prediksi Rating (Dari Data Training)")
    st.dataframe(predictions[['movie_id', 'title', 'predicted_rating']])

else:
    st.info("Silakan masukkan parameter dan klik tombol untuk melihat hasil rekomendasi.")
