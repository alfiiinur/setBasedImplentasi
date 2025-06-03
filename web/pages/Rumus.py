import streamlit as st


st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("📐 Rumus dan Evaluasi Rekomendasi")


# Mean Rating
st.markdown("## 📈 **Perhitungan Rata-Rata (Mean Rating)**")

st.latex(r"\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{u,i}")
st.latex(r"\bar{r}_i = \frac{1}{|U_i|} \sum_{u \in U_i} r_{u,i}")

st.markdown("**Keterangan:**")
st.markdown(r"$ \bar{r}_u$ : Rata-rata rating yang diberikan oleh pengguna \( u \) terhadap item \( i \).")
st.markdown(r"$\bar{r}_i $ : Rata-rata rating yang diterima oleh item \( i \) dari semua pengguna.")


st.markdown("## 📈 **Perhitungan Rata-Rata (Mean-Centered Rating)**")
st.latex(r"S_{User(u, i)} = r_{ui} - \mu_{User(u)} \quad \forall u \in \{1, \ldots, m\}")
# Menampilkan rumus dalam format LaTeX
st.latex(r"S_{Item(u, i)} = r_{ui} - \mu_{Item(i)} \quad \forall i \in \{1, \ldots, m\}")

# Menampilkan keterangan setiap simbol
st.markdown("**Keterangan:**")
st.markdown(r"$S_{Item(u,i)}$ : Mean-centered pada item *i*")
st.markdown(r"$r_{ui}$ : Rating yang diberikan oleh user *u* terhadap item *i*")
st.markdown(r"$\mu_{Item(i)}$ : Rata-rata rating pada item *i*")

# Jaccard Similarity
st.markdown("## 🔗 **Fungsi Similaritas**")

st.markdown("### 👥 **User-Based Jaccard Similarity:**")
st.latex(r"Jaccard_{user}(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}")
st.latex(r"Jaccard_{item}(i, j) = \frac{|U_i \cap U_j|}{|U_i \cup U_j|}")

st.markdown("**Keterangan:**")

st.markdown(r"$Jaccard_{user}(u, v)$ : Mengukur kesamaan antara dua item berdasarkan pengguna yang memberi rating pada keduanya.")
st.markdown(r"$Jaccard_{user}(i, j)$ : Mengukur kesamaan antara dua user yang telah memberikan rating pada item tersebut.")

st.markdown("### 📌 **Relevant Jaccard:**")
st.latex(r"""
Sim_{(u,v)}^{RJ} = \frac{1}{1 + \left(\frac{1}{|I_u \cap I_v|}\right) + \left(\frac{|\bar{I}_u|}{1 + |\bar{I}_u|}\right) + \left(\frac{|\bar{I}_v|}{1 + |\bar{I}_v|}\right)}
""")
st.latex(r"""
Sim_{(i,j)}^{RJ} = \frac{1}{1 + \left(\frac{1}{|U_i \cap U_j|}\right) + \left(\frac{|\bar{U}_i|}{1 + |\bar{U}_i|}\right) + \left(\frac{|\bar{U}_j|}{1 + |\bar{U}_j|}\right)}
""")

st.markdown("**Keterangan:**")
st.markdown(r"${I_u}, {I_v}$: Himpunan item yang telah dirating oleh user \( u \) dan \( v \)")
st.markdown(r"${I_u \cap I_v}$ : Jumlah item yang dirating oleh kedua user")
st.markdown(r"${\bar{I}_u}, {\bar{I}_v}$: Jumlah item yang dirating oleh masing-masing user \( u \) dan \( v \)")
st.markdown(r"${U_i}, {U_j} $ : Himpunan user yang memberikan rating pada item \( i \) dan \( j \)")
st.markdown(r"${\bar{U}_i}, {\bar{U}_j}$ : Jumlah user yang sama-sama memberi rating pada item \( i \) dan \( j \)")


# Top-K Neighbor
st.markdown("## 🔍 **Top-K Neighbor Selection**")
st.markdown("Pilih K tetangga terdekat berdasarkan skor similarity:")
st.latex(r"TopK(u) = \text{TopK}_u(i) = \underset{v \in U_i}{\text{argmax}} \, \text{Sim}_{\text{User}}(u, v)^{k_{\text{User}}}")

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



# DCG / IDCG / NDCG
st.markdown("## 📊 **Evaluasi NDCG (Normalized Discounted Cumulative Gain)**")

st.markdown("### 🎯 **DCG (Discounted Cumulative Gain):**")
st.latex(r"DCG_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}")

st.markdown("### 🥇 **IDCG (Ideal DCG):**")
st.latex(r"IDCG_k = \sum_{i=1}^{k} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)}")

st.markdown("### 🧮 **NDCG (Normalized DCG):**")
st.latex(r"NDCG_k = \frac{DCG_k}{IDCG_k}")
st.info("NDCG mengukur efektivitas rekomendasi dengan mempertimbangkan posisi dan relevansi item.")
