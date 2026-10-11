import argparse
import json
import os

import joblib
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "CustomerChurnModel"
PRODUCTION_DIR = "models/production"


def get_best_run_id(experiment_name: str = "telco-churn", metric: str = "pr_auc") -> str:
    """Find the run with the highest evaluation metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    best_run = runs[0]
    print(f"Best Run ID: {best_run.info.run_id} ({metric}: {best_run.data.metrics.get(metric, 0):.4f})")
    return best_run.info.run_id


def register_model(run_id: str, model_name: str = MODEL_NAME) -> str:
    """Register a trained model artifact from a run into the Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model '{model_name}' as Version {model_version.version}.")
    return model_version.version


def export_production_bundle(model_name: str = MODEL_NAME, output_dir: str = PRODUCTION_DIR):
    """
    Export the production model and preprocessor into a self-contained, OS-agnostic bundle.
    This eliminates dependency on host-specific paths (e.g. Windows C:\\ paths inside Linux Docker).
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "production")
    run = client.get_run(mv.run_id)

    model = mlflow.xgboost.load_model(f"models:/{model_name}@production")
    preprocessor = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/preprocessor")
    threshold = float(run.data.params.get("threshold", 0.35))

    os.makedirs(output_dir, exist_ok=True)
    # Save XGBoost in cross-platform JSON format
    model.save_model(os.path.join(output_dir, "model.json"))
    # Save preprocessor via joblib
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    metadata = {
        "model_name": model_name,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "threshold": threshold,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported self-contained production bundle to '{output_dir}' (Version {mv.version}).")


def set_model_stage(
    model_name: str = MODEL_NAME,
    version: str = "1",
    stage: str = "production",
):
    """
    Assign a lifecycle alias ('production', 'candidate', 'archived') to a version.
    When promoting to 'production', also exports the portable bundle for deployment.
    """
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=stage, version=version)
    client.set_model_version_tag(name=model_name, version=version, key="stage", value=stage)
    print(f"Model '{model_name}' Version {version} tagged and aliased as '@{stage}'.")

    if stage == "production":
        # 1. Remove @candidate alias so the promoted model is not its own challenger
        try:
            client.delete_registered_model_alias(name=model_name, alias="candidate")
        except mlflow.exceptions.MlflowException:
            pass

        # 2. Archive older versions
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if str(mv.version) != str(version):
                client.set_model_version_tag(name=model_name, version=mv.version, key="stage", value="archived")

        export_production_bundle(model_name=model_name)


def list_registered_models(model_name: str = MODEL_NAME):
    """List all registered versions and their current aliases."""
    client = MlflowClient()
    try:
        model = client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        print(f"No registered model named '{model_name}' found yet.")
        return

    print(f"\n--- Registered Model: {model.name} ---")
    print(f"Current Aliases: {model.aliases}")
    for mv in client.search_model_versions(f"name='{model_name}'"):
        print(f"  Version {mv.version} | Run: {mv.run_id} | Stage Tag: {mv.tags.get('stage', 'none')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage MLflow Model Registry.")
    parser.add_argument("--register-best", action="store_true", help="Find best run by PR-AUC and register it.")
    parser.add_argument("--promote", type=str, help="Version number to promote to production.")
    parser.add_argument("--export", action="store_true", help="Export current production bundle.")
    parser.add_argument("--list", action="store_true", help="List registered models and aliases.")
    args = parser.parse_args()

    if args.register_best:
        best_run_id = get_best_run_id()
        version = register_model(best_run_id)
        set_model_stage(version=version, stage="candidate")
    elif args.promote:
        set_model_stage(version=args.promote, stage="production")
    elif args.export:
        export_production_bundle()
    else:
        list_registered_models()
