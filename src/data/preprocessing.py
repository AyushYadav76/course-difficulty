import pandas as pd
import os
from src.data.ingestion import load_ednet_data
from src.utils.config_loader import load_config

def build_interaction_matrix(nrows=2_000_000):
    config = load_config()
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    df = load_ednet_data(nrows=nrows)
    questions_meta = pd.read_csv(f"{config['data']['raw_dir']}/questions.csv")
    
    # Merge metadata
    df = df.merge(questions_meta[['question_id', 'part', 'tags']], 
                  left_on='content_id', right_on='question_id', how='left')
    
    # Save
    output_path = f"{config['data']['processed_dir']}/interactions_with_meta.csv"
    df.to_csv(output_path, index=False)
    return df
