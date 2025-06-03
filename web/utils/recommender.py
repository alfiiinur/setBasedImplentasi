import pandas as pd

def load_ratings(path='utils/data/u.data'):
    columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    return pd.read_csv(path, sep='\t', names=columns)

def load_movies(path='utils/data/u.item'):
    columns = ['movie_id', 'title'] + [f'col{i}' for i in range(22)]
    df = pd.read_csv(path, sep='|', names=columns, encoding='latin-1')
    return df[['movie_id', 'title']]

def load_test_data(fold=1):
    path = f'utils/data/u{fold}.test'
    columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    return pd.read_csv(path, sep='\t', names=columns)




def get_mock_predictions(user_id):
    data = {
        'movie_id': [1, 2, 3, 4, 5],
        'title': [
            'Toy Story (1995)', 
            'Jumanji (1995)', 
            'Grumpier Old Men (1995)', 
            'Waiting to Exhale (1995)', 
            'Heat (1995)'
        ],
        'predicted_rating': [4.8, 4.5, 4.2, 3.9, 4.7]
    }
    return pd.DataFrame(data)

def get_mock_recommendations(user_id, top_n):
    data = {
        'title': [
            "The Matrix (1999)", 
            "Inception (2010)", 
            "Interstellar (2014)", 
            "The Dark Knight (2008)", 
            "Fight Club (1999)"
        ][:top_n],
        'score': [4.9, 4.8, 4.7, 4.6, 4.5][:top_n],
        'relevance': [1, 1, 1, 1, 1][:top_n],
        'ndcg': [0.98, 0.95, 0.92, 0.89, 0.85][:top_n]
    }
    return pd.DataFrame(data)
