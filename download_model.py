
import pickle
import logging
import os
import sys
import mlflow

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

MODEL_VERSION_FILE = ".model-version"
MODEL_NAME = "wine-quality"
OUTPUT_FILE = "wine_quality_model.pkl"



def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def main() -> None:
    check_env_vars()
    run_id = "8a08729defcc419c80bc525729c2f5cb"

    # Switch the URI prefix to runs:/ so it targets artifacts directly
    # The structure is: runs:/<run_id>/<artifact_path_where_model_is_saved>
    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Downloading model from artifact storage: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
