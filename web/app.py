import streamlit as st

st.title("🎬 Sistem Rekomendasi")
st.markdown("""
Selamat datang di **Sistem Rekomendasi Film** dengan menggunakan dataset MovieLens 100K.  
Aplikasi ini menggunakan model berbasis Hybrid dengan kombinasi User-Based (UBCF) dan Item-Based (IBCF) serta untuk evaluasi menggunakan metrik **NDCG**  

**Fitur utama:**
- Sistem Rekomendasi Berbasis Collaborative Filtering
- Menggnakan Pendekatan Hybrid (User-Based dan Item-Based)
- Pilihan metode Similarity : Jaccard, Relevant Jaccard (RJ)
- Evaluasi Menggunakan Metrik Evaluasi NDCG(Normalized Discounted Cumulative Gain)
- Tampilan Home, Rumus, dan Prediksi
""")

st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif", width=400)
st.success("Mulai dari tab *🔍 Similarity* di sidebar!")
