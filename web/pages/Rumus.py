import streamlit as st


st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("📐 Rumus dan Evaluasi Rekomendasi")

# DCG / IDCG / NDCG
st.markdown("## 📊 **Evaluasi NDCG (Normalized Discounted Cumulative Gain)**")

st.markdown("### 🎯 **DCG (Discounted Cumulative Gain):**")
st.latex(r"DCG_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}")

st.markdown("### 🥇 **IDCG (Ideal DCG):**")
st.latex(r"IDCG_k = \sum_{i=1}^{k} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)}")

st.markdown("### 🧮 **NDCG (Normalized DCG):**")
st.latex(r"NDCG_k = \frac{DCG_k}{IDCG_k}")
st.info("NDCG mengukur efektivitas rekomendasi dengan mempertimbangkan posisi dan relevansi item.")

# Mean Rating
st.markdown("## 📈 **Perhitungan Rata-Rata (Mean) dan Mean-Centered**")

st.markdown("### 👤 **Mean Rating (User-Based):**")
st.latex(r"\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{u,i}")
st.info("Rata-rata rating yang diberikan oleh pengguna \( u \) terhadap semua item yang ia beri rating.")

st.markdown("### 📦 **Mean Rating (Item-Based):**")
st.latex(r"\bar{r}_i = \frac{1}{|U_i|} \sum_{u \in U_i} r_{u,i}")
st.info("Rata-rata rating yang diterima oleh item \( i \) dari semua pengguna.")

st.markdown("### 🔄 **Mean-Centered Rating:**")
st.latex(r"r'_{u,i} = r_{u,i} - \bar{r}_u")
st.latex(r"r'_{u,i} = r_{u,i} - \bar{r}_i")
st.info("Mean-centered digunakan untuk menyesuaikan rating terhadap rata-rata masing-masing user atau item.")

# Jaccard Similarity
st.markdown("## 🔗 **Jaccard Similarity**")

st.markdown("### 👥 **User-Based Jaccard Similarity:**")
st.latex(r"Jaccard_{user}(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}")
st.info("Mengukur kesamaan antara dua pengguna berdasarkan item yang sama-sama mereka beri rating.")

st.markdown("### 🎬 **Item-Based Jaccard Similarity:**")
st.latex(r"Jaccard_{item}(i, j) = \frac{|U_i \cap U_j|}{|U_i \cup U_j|}")
st.info("Mengukur kesamaan antara dua item berdasarkan pengguna yang memberi rating pada keduanya.")

st.markdown("### 📌 **Relevant Jaccard:**")
st.latex(r"Jaccard_{rel}(u, v) = \frac{|I_u^{rel} \cap I_v^{rel}|}{|I_u^{rel} \cup I_v^{rel}|}")
st.info("Versi Jaccard yang hanya memperhitungkan item relevan (misalnya rating >= threshold).")

# Top-K Neighbor
st.markdown("## 🔍 **Top-K Neighbor Selection**")
st.markdown("Pilih K tetangga terdekat berdasarkan skor similarity:")
st.latex(r"TopK(u) = \text{arg top-k}_{v \neq u}(similarity(u, v))")

# Prediksi
st.markdown("## 🎯 **Prediksi Rating**")

st.markdown("### 👤 **User-Based CF Prediction:**")
st.latex(r"\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N_k(u)} |sim(u,v)|}")

st.markdown("### 🎬 **Item-Based CF Prediction:**")
st.latex(r"\hat{r}_{u,i} = \bar{r}_i + \frac{\sum_{j \in N_k(i)} sim(i,j) \cdot (r_{u,j} - \bar{r}_j)}{\sum_{j \in N_k(i)} |sim(i,j)|}")

# Hybrid
st.markdown("## ♻️ **Hybrid Recommendation (Linear Combination)**")
st.latex(r"\hat{r}_{u,i}^{hybrid} = \alpha \cdot \hat{r}_{u,i}^{user} + (1 - \alpha) \cdot \hat{r}_{u,i}^{item}")
st.info("Menggabungkan prediksi dari user-based dan item-based dengan bobot \( \alpha \).")

# Top-N Recommendation
st.markdown("## 🧾 **Top-N Recommendation**")
st.markdown("Urutkan prediksi tertinggi untuk setiap user, lalu pilih \( N \) item:")
st.latex(r"TopN(u) = \text{Top-}N(\hat{r}_{u,i} \text{ untuk semua } i \in I)")
st.info("Digunakan untuk menghasilkan daftar rekomendasi terbaik bagi setiap pengguna.")
