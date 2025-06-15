import pandas as pd
import joblib

def load_test_data():
    return pd.read_csv("utils/data/u1.test", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

def load_training_data():
    return pd.read_csv("utils/data/u1.base", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

def load_item_data():
    return pd.read_csv("utils/data/u.item", sep="|", names=["movie_id", "title"], encoding='latin-1', usecols=[0, 1])

def load_topn(method):
    return joblib.load(f"utils/model/topN/{method}/5_10_1.joblib")

def load_ndcg(method):
    return joblib.load(f"utils/model/ndcg/{method}/5_10_1_ndcg.joblib")

def get_intersection(recommended_df, ground_truth_df):
    return pd.merge(recommended_df, ground_truth_df[['movie_id']], on='movie_id', how='inner')


def generate_recommendations(user_id, method, k=20):
    item_data = load_item_data()
    topn = load_topn(method)
    ndcg_df = load_ndcg(method)

    try:
        topn_ids = topn[user_id - 1][:k]
    except IndexError:
        topn_ids = []

    recommended = item_data[item_data["movie_id"].isin(topn_ids)].copy()

    try:
        ndcg_score = ndcg_df.loc[f"User-{user_id}", f"NDCG@{k}"]
    except:
        ndcg_score = 0.0

    return recommended, ndcg_score
