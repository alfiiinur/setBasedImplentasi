import streamlit as st

from utils.recommender import (
    get_mock_ground_truth,
    get_mock_predictions,
    get_mock_recommendations
)



st.title("🔍 Rekomendasi Berdasarkan Similarity")

# Form input
similarity_method = st.selectbox("Pilih Metode Similarity", ["Cosine", "Pearson", "Jaccard"])
top_n = st.selectbox("Top-N Rekomendasi", [5, 10, 15, 20], index=1)
user_id = st.number_input("Masukkan User ID", min_value=1, max_value=1000, step=1)

if st.button("🎯 Proses Rekomendasi"):
    # Get mock data for ground truth, predictions, and recommendations
    ground_truth = get_mock_ground_truth(user_id)
    predictions = get_mock_predictions(user_id)
    recommendations = get_mock_recommendations(user_id, top_n)

    # Calculate NDCG (for demonstration, you can replace it with actual calculation)
    # Here, we're just mocking the NDCG value for illustration purposes
    ndcg_value = 0.85  # Replace this with actual NDCG calculation logic

    # Display method, top N, user ID, and NDCG
    st.markdown(f"### 🔎 **Informasi Pengguna dan Hasil**")
    st.markdown(f"- **Metode Similarity yang digunakan:** {similarity_method}")
    st.markdown(f"- **Top-N Rekomendasi:** {top_n}")
    st.markdown(f"- **User ID:** {user_id}")
    st.markdown(f"- **Hasil NDCG:** {ndcg_value:.2f}")
    
    # Bagi jadi 3 kolom untuk menampilkan hasil data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📘 Ground Truth")
        st.dataframe(ground_truth)

    with col2:
        st.markdown("### 📊 Prediksi")
        st.dataframe(predictions)

    with col3:
        st.markdown(f"### 🎬 Rekomendasi (Top-{top_n})")
        st.dataframe(recommendations)

else:
    st.info("Masukkan parameter lalu klik tombol untuk mendapatkan rekomendasi.")
