import os
from mlflow.server import _run_server

if __name__ == "__main__":
    # Set MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "postgresql://mlflow:mlflow@db/mlflow"

    # Run MLflow server
    _run_server(
        host="0.0.0.0",
        port=5000,
        default_artifact_root="../ml",
        serve_artifacts=True
    )
