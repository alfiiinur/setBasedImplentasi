import streamlit as st

st.title("🎬 Sistem Rekomendasi Film")
st.markdown("""
Selamat datang di **Sistem Rekomendasi Film** berbasis dataset MovieLens 100K.  
Aplikasi ini menggunakan model prediksi dan evaluasi menggunakan metrik **NDCG**  
untuk memberikan rekomendasi film terbaik kepada pengguna.

**Fitur utama:**
- Evaluasi berdasarkan NDCG
- Pilihan metode Similarity (Cosine, Pearson, Jaccard)
- Tampilan riwayat data, prediksi, dan hasil rekomendasi
""")

st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif", width=400)
st.success("Mulai dari tab *🔍 Similarity* di sidebar!")
