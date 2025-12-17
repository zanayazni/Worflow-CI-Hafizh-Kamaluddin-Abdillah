import argparse
import os
from pathlib import Path
from contextlib import nullcontext

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="CI - Seattle Weather Best RandomForest",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    if os.getenv("MLFLOW_RUN_ID") or os.getenv("MLFLOW_EXPERIMENT_ID"):
        print(
            "[INFO] Detected MLflow Projects run; skipping mlflow.set_experiment(...)"
        )
    else:
        mlflow.set_experiment(args.experiment_name)

    print(f"[INFO] Using MLflow tracking URI: {tracking_uri}")

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "seattle-weather_preprocessing.csv"

    df = pd.read_csv(data_path)

    X = df.drop(columns=["weather"])
    y = df["weather"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_params = {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    }

    run_name = "CI_Best_RandomForest_SeattleWeather"

    active = mlflow.active_run()
    if active is not None:
        run_context = nullcontext()
    else:
        run_context = mlflow.start_run(run_name=run_name)

    with run_context:
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("Failed to start or resume an MLflow run.")

        run_id = run.info.run_id
        print(f"[INFO] MLflow Run ID: {run_id}")

        # Simpan run_id untuk CI
        run_id_path = base_dir / "last_run_id.txt"
        run_id_path.write_text(run_id, encoding="utf-8")

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred)

        # Logging ke MLflow
        mlflow.log_param("model_name", "RandomForest_Best_Params_CI")
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)

        mlflow.sklearn.log_model(model, artifact_path="model")

        cm_path = base_dir / "confusion_matrix.txt"
        cm_path.write_text(str(cm), encoding="utf-8")
        mlflow.log_artifact(cm_path)

        report_path = base_dir / "classification_report_rf_ci.txt"
        report_path.write_text(
            classification_report(y_test, y_pred),
            encoding="utf-8",
        )
        mlflow.log_artifact(report_path)

        mlflow.log_artifact(run_id_path)

        print("=== CI TRAINING FINISHED ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-Score : {f1:.4f}")


if __name__ == "__main__":
    main()
