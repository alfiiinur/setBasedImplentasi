import streamlit as st
import pandas as pd
from utils.recommender import (
    load_test_data,
    load_training_data,
    load_item_data,
    generate_recommendations,
    get_intersection
)




# Fungsi highlight baris yang cocok
def highlight_matches(row, match_ids):
    if row['movie_id'] in match_ids:
        return ['background-color: #d4edda'] * len(row)  # Hijau muda untuk match
    else:
        return [''] * len(row)

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("🔍 Rekomendasi Film Berdasarkan Fungsi Similaritas")

# ========================
# Form Input
# ========================
similarity_method = st.selectbox("📌 Pilih Fungsi Similaritas", ["Jaccard", "RJ"])
top_n = st.selectbox("🔢 Top-N Rekomendasi", list(range(1, 21)), index=4)  # default 5
user_id = st.number_input("👤 Masukkan User ID", min_value=1, max_value=459, step=1)

# ========================
# Proses Rekomendasi
# ========================
if st.button("🎯 Proses Rekomendasi"):
    # Load data dan rekomendasi
    test_data = load_test_data()
    training_data = load_training_data()
    item_data = load_item_data()

    recommended_df, ndcg_value = generate_recommendations(user_id, similarity_method, k=top_n)

    # Ground truth dari test set
    gt_test = test_data[test_data['user_id'] == user_id].merge(item_data, on="movie_id", how="left")

    # Data training user
    gt_train = training_data[training_data['user_id'] == user_id].merge(item_data, on="movie_id", how="left")

    # Irisan antara rekomendasi dan ground truth
    intersection = get_intersection(recommended_df, gt_test)

    # ========================
    # Informasi Ringkasan
    # ========================
    st.markdown("### 📝 **Ringkasan Evaluasi**")
    st.markdown(f"- **Metode Similarity:** `{similarity_method}`")
    st.markdown(f"- **Top-N:** `{top_n}`")
    st.markdown(f"- **User ID:** `{user_id}`")
    st.markdown(f"- **NDCG@{top_n}:** `{ndcg_value:.4f}`")
    st.markdown(f"- **Jumlah Interseksi (benar di Top-{top_n}):** `{len(intersection)}` item")

    # ========================
    # Tampilkan Data
    # ========================
    col1, col2, col3 = st.columns(3)



    # def add_index_column(df):
    #     df = df.reset_index(drop=True)
    #     df.insert(0, 'No', df.index)
    #     return df


    with col1:
        st.subheader("🎯 Ground Truth (Test Set)")
        # st.divider()
        st.text("Data test dari target user yang didapatkan dari dataset dan untuk digunakan di dalam evaluasi hasil rekomendasi metode. Data Test didapatkan  sebanyak `20%` dari data training.")

        if gt_test.empty:
            st.warning("User tidak memiliki data test.")
        else:
            st.dataframe(gt_test[['movie_id', 'title', 'rating']])

    with col2:
        st.subheader("🧠 Data Training")
        # st.divider()
        st.text(f"Data training yaitu data item yang telah diberikan rating oleh target user yang didapatkan dari user -  {user_id} ")
        if gt_train.empty:
            st.warning("User tidak memiliki data training.")
        else:
            st.dataframe(gt_train[['movie_id', 'title', 'rating']])


    with col3:
        st.subheader("📢 Rekomendasi")
        # st.divider()
        st.text(f"Data rekomendasi yang cocok dengan data test (interseksi) sebanyak {len(intersection)} item.")

        if recommended_df.empty:
            st.warning("Tidak ada rekomendasi.")
        else:
            st.markdown(f"Jumlah item cocok dengan ground truth: `{len(intersection)}` dari Top-{top_n}`")

            match_ids = intersection["movie_id"].tolist()
            styled_recs = recommended_df[['movie_id', 'top_n', 'title']].style.apply(highlight_matches, axis=1, match_ids=match_ids)
            st.dataframe(styled_recs)

            if not intersection.empty:
                st.markdown("### ✅ Irisan dengan Ground Truth")
                st.dataframe(intersection[['movie_id', 'top_n', 'title']])

else:
    st.info("Silakan masukkan parameter dan klik tombol untuk melihat hasil rekomendasi.")
