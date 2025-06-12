import pandas as pd

def load_test_data():
    return pd.read_csv("utils/data/u1.test", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

def load_training_data():
    return pd.read_csv("utils/data/u1.base", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

def load_item_data():
    return pd.read_csv("utils/data/u.item", sep="|", names=["movie_id", "title"], encoding='latin-1', usecols=[0, 1])

def generate_recommendations(user_id, top_n, similarity_method):
    # Dummy rekomendasi, sebaiknya diganti dengan model prediksi
    item_data = load_item_data()
    recs = item_data.sample(n=top_n).copy()
    recs['movie_id'] = recs['movie_id'].astype(int)
    return recs
