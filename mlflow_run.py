import mlflow
import pandas as pd
from src.data.preprocessing import build_interaction_matrix
from src.data.feature_engineering import build_question_features
from src.models.student.irt import IRT2PL
from src.models.difficulty.clustering import cluster_questions
from src.utils.config_loader import load_config

def main():
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "irt_samples": config['model']['irt_samples'],
            "n_clusters": config['model']['n_clusters']
        })

        # Build data
        interactions = build_interaction_matrix(nrows=500_000)  # Reduce for demo
        features = build_question_features(interactions)

        # Fit IRT (optional: skip if too slow)
        # irt_model = IRT2PL(samples=config['model']['irt_samples'], tune=config['model']['irt_tune'])
        # irt_model.fit(interactions)
        # irt_diff = irt_model.get_item_difficulty()
        # features = features.merge(irt_diff, on='content_id', how='left')
        # irt_model.save("models/irt_model.joblib")

        # Cluster
        clustered_df, sil_score = cluster_questions(features)
        mlflow.log_metric("silhouette_score", sil_score)

        # Save artifacts
        mlflow.log_artifact("output/final_question_difficulty.csv")
        mlflow.log_artifact("models/scaler.joblib")
        mlflow.log_artifact("models/gmm_model.joblib")

if __name__ == "__main__":
    main()
