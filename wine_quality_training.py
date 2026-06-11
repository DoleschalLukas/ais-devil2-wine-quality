import mlflow
import os
import json
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import pandas
import pandas as pd
import numpy as np


DATA_FILE = "data/winequality.parquet"


REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        sys.exit(1)

def train_random_forest_classifier(data_file: str) -> tuple[RandomForestClassifier, dict]:
  check_env_vars()
  data = pd.read_parquet(data_file)
  X = data[['fixed_acidity', 'citric_acid', 'residual_sugar', 'volatile_acidity', 'chlorides']]
  y = data['quality']

  # As the dataset is imbalanced, stratify=y will ensure that the split maintains the proportion of classes
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Use class_weight='balanced' to handle class imbalance
  clf = RandomForestClassifier(class_weight='balanced', random_state=42)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)
  print(report)

  return (clf, report)

def train_random_forest_classifier_v2(data_file: str) -> tuple[RandomForestClassifier, dict]:
  data = pandas.read_parquet(data_file)
  X = data[['fixed_acidity', 'citric_acid', 'residual_sugar', 'volatile_acidity', 'chlorides']]
  y = data['quality']

  # As the dataset is imbalanced, stratify=y will ensure that the split maintains the proportion of classes
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = Pipeline([
    # Use class_weight='balanced' to handle class imbalance
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
  ])
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def train_logistic_regression_classifier(data_file: str) -> tuple[LogisticRegression, dict]:
  data = pandas.read_parquet(data_file)
  X = data[['fixed_acidity', 'citric_acid', 'residual_sugar', 'volatile_acidity', 'chlorides']]
  y = data['quality']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = LogisticRegression(class_weight='balanced', random_state=42,
                           max_iter=1000)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def check_env_vars() -> None:
  missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
  if missing:
    sys.exit(1)

def train_model(model_type: str):
    check_env_vars()
    mlflow.set_experiment(model_type)
    mlflow.autolog()
    with mlflow.start_run():
      if model_type == 'random_forest':
        model, metadata = train_random_forest_classifier(DATA_FILE)
      elif model_type == 'random_forest_v2':
        model, metadata = train_random_forest_classifier_v2(DATA_FILE)
      elif model_type == 'logistic_regression':
        model, metadata = train_logistic_regression_classifier(DATA_FILE)

      print("Available keys in metadata:", list(metadata.keys()))
      print("Full metadata structure:", metadata)

      # Dynamically log metrics for each wine quality class
      for class_label, metrics in metadata.items():
        # Check if the value is a dictionary (like the class scores or averages)
        if isinstance(metrics, dict):
          for metric_name, metric_value in metrics.items():
            # This logs clean names like: class_5_precision, macro_avg_f1-score
            clean_label = class_label.replace(" ", "_")
            mlflow.log_metric(f"{clean_label}_{metric_name}", metric_value)
        else:
          # This catches the global 'accuracy' metric which is a single float
          mlflow.log_metric(class_label, metrics)
      os.makedirs("models", exist_ok=True)

      model_output_file = f"models/{model_type}.pkl"
      with open(model_output_file, "wb") as f:
        pickle.dump(model, f)

      metadata_output_file = f"models/{model_type}.metadata.json"
      with open(metadata_output_file, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

      mlflow.log_artifact(metadata_output_file)

    valid_model_types = ['random_forest', 'random_forest_v2', 'logistic_regression']
    if model_type not in valid_model_types:
      raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {', '.join(valid_model_types)}")

    if model_type == 'random_forest':
      model, metadata = train_random_forest_classifier(DATA_FILE)
    elif model_type == 'random_forest_v2':
      model, metadata = train_random_forest_classifier_v2(DATA_FILE)
      print("v2")
    elif model_type == 'logistic_regression':
      model, metadata = train_logistic_regression_classifier(DATA_FILE)

if __name__ == "__main__":
  model_type = sys.argv[1]  # random_forest, random_forest_v2, logistic_regression
  train_model(model_type)