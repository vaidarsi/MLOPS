import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow

# Sample dataset
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5],
    "Scores": [10, 20, 30, 40, 50]
})

X = data[["Hours"]]
y = data["Scores"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow tracking
mlflow.start_run()

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

mlflow.log_param("model", "LinearRegression")
mlflow.log_metric("mse", mse)

mlflow.end_run()

print("Model trained successfully!")
print("Predictions:", predictions)
print("MSE:", mse)