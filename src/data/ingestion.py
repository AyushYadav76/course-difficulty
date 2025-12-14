import pandas as pd
from src.utils.config_loader import load_config

def load_ednet_data(nrows=None):
    config = load_config()
    path = f"{config['data']['raw_dir']}/train.csv"
    usecols = ['user_id', 'content_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time']
    dtype = {
        'user_id': 'int32',
        'content_id': 'int32',
        'content_type_id': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32'
    }
    df = pd.read_csv(path, nrows=nrows, usecols=usecols, dtype=dtype)
    return df[df['content_type_id'] == 0]
