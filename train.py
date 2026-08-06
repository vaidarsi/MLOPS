# ==========================================
# Training Script (MLflow + DagsHub + DVC)
# ==========================================

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# DagsHub + MLflow Configuration
# ==========================================

os.environ["MLFLOW_TRACKING_USERNAME"] = "vaidarsi"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d297149d179a4015b446c442b05d4d0e52672495"

mlflow.set_tracking_uri("https://dagshub.com/vaidarsi/MLOPS_PROJECT.mlflow")
mlflow.set_experiment("DVC_MLflow_Pipeline")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/processed.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    # ======================================
    # Log Parameters
    # ======================================

    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 5)

    # ======================================
    # Log Metrics
    # ======================================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # ======================================
    # Log Model to MLflow
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # ======================================
    # Save Model for DVC
    # ======================================

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    # ======================================
    # Print Results
    # ======================================

    print("=" * 50)
    print("Training Completed Successfully")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("=" * 50)