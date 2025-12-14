import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

class IRT2PL:
    def __init__(self, samples=1000, tune=1000):
        self.samples = samples
        self.tune = tune
        self.model = None
        self.trace = None
        self.user_encoder = None
        self.item_encoder = None

    def fit(self, df):
        # Encode IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        users = self.user_encoder.fit_transform(df['user_id'])
        items = self.item_encoder.fit_transform(df['content_id'])
        responses = df['answered_correctly'].values

        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)

        with pm.Model() as model:
            ability = pm.Normal("ability", mu=0, sigma=1, shape=n_users)
            difficulty = pm.Normal("difficulty", mu=0, sigma=1, shape=n_items)
            discrimination = pm.HalfNormal("discrimination", sigma=1, shape=n_items)

            logits = discrimination[items] * (ability[users] - difficulty[items])
            pm.Bernoulli("obs", logit_p=logits, observed=responses)

            self.trace = pm.sample(self.samples, tune=self.tune, target_accept=0.9, return_inferencedata=True)
            self.model = model

        return self

    def get_item_difficulty(self):
        difficulty_samples = az.extract(self.trace, var_names=["difficulty"])
        difficulty_mean = difficulty_samples.mean(dim="sample").values
        item_ids = self.item_encoder.inverse_transform(np.arange(len(difficulty_mean)))
        return pd.DataFrame({
            'content_id': item_ids,
            'irt_difficulty': difficulty_mean
        })

    def save(self, path):
        joblib.dump({
            'trace': self.trace,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.trace = data['trace']
        instance.user_encoder = data['user_encoder']
        instance.item_encoder = data['item_encoder']
        return instance
