import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os
from src.utils.config_loader import load_config

def cluster_questions(df):
    config = load_config()
    os.makedirs("models", exist_ok=True)
    
    # Use IRT difficulty if available, else accuracy
    if 'irt_difficulty' in df.columns:
        features = ['irt_difficulty', 'avg_time', 'retry_rate']
    else:
        features = ['accuracy', 'avg_time', 'retry_rate']
    
    X = df[features].copy().fillna(df[features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=config['model']['n_clusters'], random_state=42)
    clusters = gmm.fit_predict(X_scaled)
    
    # Label by difficulty
    df['cluster'] = clusters
    if 'irt_difficulty' in df.columns:
        cluster_diff = df.groupby('cluster')['irt_difficulty'].mean()
        order = cluster_diff.sort_values(ascending=False).index  # Higher IRT = harder
    else:
        cluster_acc = df.groupby('cluster')['accuracy'].mean()
        order = cluster_acc.sort_values().index  # Lower accuracy = harder
    
    level_map = {order[0]: 'Hard', order[1]: 'Medium', order[2]: 'Easy'}
    df['difficulty_level'] = df['cluster'].map(level_map)
    
    # Save
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(gmm, "models/gmm_model.joblib")
    df.to_csv("output/final_question_difficulty.csv", index=False)
    
    sil_score = silhouette_score(X_scaled, clusters)
    return df, sil_score
