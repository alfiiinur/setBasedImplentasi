import streamlit as st
import pandas as pd
import joblib

# ------------------ Load Files ------------------ #
@st.cache
def load_data():
    # Load topN list of list dan konversi ke dict
    topN_list = joblib.load("utils/model/topN/RJ/5_10_1.joblib")
    ndcg_df = joblib.load("utils/model/ndcg/RJ/5_10_1_ndcg.joblib")  # DataFrame

    train_df = pd.read_csv("utils/data/u1.base", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    test_df = pd.read_csv("utils/data/u1.test", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    items_df = pd.read_csv("utils/data/u.item", sep="|", encoding='latin-1', usecols=[0, 1], names=["item_id", "title"])

    # Buat topN dict
    unique_user_ids = sorted(train_df['user_id'].unique())[:len(topN_list)]
    topN_model = {uid: recs for uid, recs in zip(unique_user_ids, topN_list)}

    # Ambil NDCG@20 dari DataFrame
    ndcg_at_20 = {}
    for idx, row in ndcg_df.iterrows():
        if idx.startswith("User-"):
            user_number = int(idx.split("-")[1])
            ndcg_at_20[user_number] = row["NDCG@20"] if "NDCG@20" in row else row.iloc[-1]  # pakai kolom terakhir

    return topN_model, ndcg_at_20, train_df, test_df, items_df

topN_model, ndcg_values, train_df, test_df, items_df = load_data()

# ------------------ Streamlit App ------------------ #
st.title("🎬 MovieLens Recommender - RJ TopN@20")

user_ids = sorted(topN_model.keys())
user_id = st.sidebar.selectbox("Pilih User ID", user_ids)

# ------------------ Data untuk user ------------------ #
recommended_items = topN_model.get(user_id, [])
gt_items = set(test_df[test_df['user_id'] == user_id]['item_id'])
ndcg = ndcg_values.get(user_id, 0.0)
user_train_items = train_df[train_df['user_id'] == user_id]['item_id'].tolist()

# ------------------ Info Panel ------------------ #
st.markdown(f"### 👤 User ID: `{user_id}`")
st.write(f"📦 **Jumlah Interaksi di Test Set (GT)**: `{len(gt_items)}`")
st.write(f"📊 **NDCG@20**: `{ndcg:.4f}`")
st.write(f"🧠 **Jumlah Item Dikenal di Training**: `{len(user_train_items)}`")

# ------------------ Tabel Rekomendasi ------------------ #
st.markdown("### ⭐ Rekomendasi TopN@20")

reco_df = pd.DataFrame({'item_id': recommended_items})
reco_df = reco_df.merge(items_df, on='item_id', how='left')
reco_df['GT Match'] = reco_df['item_id'].apply(lambda x: '✔️' if x in gt_items else '')

def highlight_match(val):
    return 'background-color: lightgreen' if val == '✔️' else ''

st.dataframe(reco_df.style.applymap(highlight_match, subset=['GT Match']))

# ------------------ Tabel Ground Truth ------------------ #
st.markdown("### ✅ Ground Truth Item di Test Set")
gt_df = pd.DataFrame({'item_id': list(gt_items)})
gt_df = gt_df.merge(items_df, on='item_id', how='left')
st.dataframe(gt_df)

# ------------------ Tabel Training User ------------------ #
st.markdown("### 🧾 Item Pernah Dilihat (Training Set)")
train_items_df = train_df[train_df['user_id'] == user_id][['item_id']]
train_items_df = train_items_df.merge(items_df, on='item_id', how='left')
st.dataframe(train_items_df)

# ------------------ Tabel Semua Item ------------------ #
if st.checkbox("📚 Tampilkan Semua Item di Database"):
    st.markdown("### 🎞️ Semua Item")
    st.dataframe(items_df)
