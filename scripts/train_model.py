# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Update the import
from sklearn.metrics import mean_squared_error
import joblib
import sys
import mlflow

def train_random_forest(train_data_path, test_data_path, model_output_path):
    # Load DVC-managed data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Specify the columns to be used as features
    feature_columns = ['year', 'month', 'day', 'hour', 'Machine_ID', 'Sensor_ID']

    # Assume 'Reading' is the column you want to predict
    X_train = train_data[feature_columns]
    y_train = train_data['Reading']
    
    X_test = test_data[feature_columns]
    y_test = test_data['Reading']

    # Train a Random Forest regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error (adjust as needed)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mean_squared_error", mse)

    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "model")

   
    # Register the model in the Model Registry
    model_name="sensor_reading_predictor"
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

    # Save the trained model
    joblib.dump(model, model_output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <train_data_path> <test_data_path> <model_output_path>")
        sys.exit(1)

    train_data_path, test_data_path, model_output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Start MLflow run
    with mlflow.start_run():
        train_random_forest(train_data_path, test_data_path, model_output_path)
