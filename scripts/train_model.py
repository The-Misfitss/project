# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        # Add other hyperparameters as needed
    }

    # Create Random Forest regression model
    rf = RandomForestRegressor(random_state=42)

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Log best hyperparameters and metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mean_squared_error", mse)

    # Log the trained model to MLflow
    mlflow.sklearn.log_model(best_model, "model")
    model_name = "sensor_reading_predictor"
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

    # Save the trained model
    joblib.dump(best_model, model_output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <train_data_path> <test_data_path> <model_output_path>")
        sys.exit(1)

    train_data_path, test_data_path, model_output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Start MLflow run
    with mlflow.start_run():
        train_random_forest(train_data_path, test_data_path, model_output_path)
