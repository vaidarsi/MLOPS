import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# -------------------------
# Configure MLflow
# -------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine_Hyperparameter_Tuning")

# -------------------------
# Load Dataset
# -------------------------
data = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.4,
    random_state=42
)

#  Add noise to make problem harder (so accuracy varies)
X_train = X_train + np.random.normal(0, 0.3, X_train.shape)
X_test = X_test + np.random.normal(0, 0.3, X_test.shape)

# -------------------------
# Hyperparameter Grid
# -------------------------
param_grid = {
    "n_estimators": [1, 5, 10, 50],
    "max_depth": [1, 2, 3, 5, None]
}

grid = list(ParameterGrid(param_grid))

best_accuracy = 0
best_model = None

# -------------------------
# Run Experiments
# -------------------------
for i, params in enumerate(grid):

    with mlflow.start_run(run_name=f"Run_{i+1}"):

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        #  Use NEW API (name instead of artifact_path)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"Run {i+1} | {params} | Accuracy: {acc}")

        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

# -------------------------
# Log BEST model to registry
# -------------------------
with mlflow.start_run(run_name="Best_Model"):

    mlflow.log_metric("best_accuracy", best_accuracy)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="Wine_RF_Model"
    )

print("\n Best Accuracy:", best_accuracy)

# -------------------------
# Promote Best Model
# -------------------------
client = MlflowClient()

versions = client.search_model_versions("name='Wine_RF_Model'")

# Get latest version (best model we just logged)
latest_version = max([int(v.version) for v in versions])

client.set_registered_model_alias(
    name="Wine_RF_Model",
    alias="production",
    version=latest_version
)

print(f"\n Best model (Version {latest_version}) promoted to PRODUCTION")