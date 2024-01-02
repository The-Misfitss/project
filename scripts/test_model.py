import requests
import json
import pandas as pd
from sklearn.metrics import mean_squared_error

def calculate_mse(data, target, url):
    # Convert data and target to JSON format
    payload = json.dumps({"data": data, "target": target})

    # Set the headers for the POST request
    headers = {
        'Content-Type': 'application/json'
    }

    # Send POST request to the Flask app
    response = requests.post(url, headers=headers, data=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get predictions from the response
        predictions = response.json()['predictions']
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(target, predictions)
        return mse
    else:
        # If the request fails, return None or handle the error accordingly
        print("Request failed. Status code:", response.status_code)
        return None

def main():
    # Load data from the CSV file (adjust file path as needed)
    data = pd.read_csv('data/processed/new_data.csv')
    
    # Specify the columns to be used as features
    feature_columns = ['year', 'month', 'day', 'hour', 'Machine_ID', 'Sensor_ID']

    # Separate features and target variable
    target_column = 'Reading'
    features = data[feature_columns]
    target = data[target_column]

    # Update with actual Flask app URL
    flask_url = 'http://localhost:5000/batch-predict'

    # Calculate MSE using the function
    mse_result = calculate_mse(features.values.tolist(), target.tolist(), flask_url)
    if mse_result is not None:
        print(f"Mean Squared Error (MSE): {mse_result:.4f}")
        # Write MSE to a file named 'mse_result.txt'
        with open('mse_result.txt', 'w') as mse_file:
            mse_file.write(str(mse_result))
    else:
        with open('mse_result.txt', 'w') as mse_file:
            mse_file.write(str(-1.0))

if __name__ == "__main__":
    main()
