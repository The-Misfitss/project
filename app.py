from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained machine learning model here
model = joblib.load('model/random_forest_model.joblib')

# Define route to render HTML form
@app.route('/')
def index():
    return render_template('index.html')  # Render HTML template

# Define route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from HTML form
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    hour = int(request.form['hour'])
    machine_id = int(request.form['Machine_ID'])
    sensor_id = int(request.form['Sensor_ID'])

    # Perform prediction based on user input features
    prediction = model.predict([[year, month, day, hour, machine_id, sensor_id]])  # Replace with appropriate features

    return jsonify({'prediction': prediction.tolist()})

# Define a route for testing predictions on multiple rows of features
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    # Get features as JSON data from request
    features_json = request.json  # JSON should contain an array of feature rows

    # Extract features and perform batch prediction
    feature_rows = features_json['data']  # Assuming the JSON key is 'data' containing feature rows
    predictions = []  # List to store predictions

    for row in feature_rows:
        prediction = model.predict([row])  # Make prediction for each feature row
        predictions.append(prediction.tolist())  # Append prediction to the list

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
