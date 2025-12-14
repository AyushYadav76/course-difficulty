import pandas as pd
import numpy as np
import os
from src.utils.config_loader import load_config

def build_question_features(interactions_df):
    config = load_config()
    os.makedirs(config['data']['features_dir'], exist_ok=True)
    
    # Basic stats
    stats = interactions_df.groupby('content_id').agg(
        total_attempts=('answered_correctly', 'count'),
        correct_attempts=('answered_correctly', 'sum'),
        avg_time=('prior_question_elapsed_time', 'mean'),
        std_time=('prior_question_elapsed_time', 'std')
    ).reset_index()
    
    stats['accuracy'] = stats['correct_attempts'] / stats['total_attempts']
    stats['failure_rate'] = 1 - stats['accuracy']
    stats['std_time'] = stats['std_time'].fillna(0)
    
    # Retry rate
    user_attempts = interactions_df.groupby(['content_id', 'user_id']).size().reset_index(name='attempts')
    retry_rate = (user_attempts['attempts'] > 1).groupby(user_attempts['content_id']).mean()
    stats = stats.merge(retry_rate.rename('retry_rate'), on='content_id', how='left')
    stats['retry_rate'] = stats['retry_rate'].fillna(0)
    
    # Skill-level difficulty (simplified)
    stats['part'] = interactions_df.groupby('content_id')['part'].first().values
    stats['part_difficulty'] = stats['part'].map({1:0.3, 2:0.35, 3:0.4, 4:0.45, 5:0.6, 6:0.7, 7:0.75})
    
    output_path = f"{config['data']['features_dir']}/question_features.csv"
    stats.to_csv(output_path, index=False)
    return stats
