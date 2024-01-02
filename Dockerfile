FROM python:3.8

# Set working directory
WORKDIR /app

# Copy requirements.txt to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and Flask application code
COPY app.py .
COPY model/random_forest_model.joblib ./model/

# Copy HTML file
COPY templates/index.html ./templates/

# Expose port
EXPOSE 5000

# Set entry point to start the Flask app
CMD ["python", "app.py"]

# # Set entry point to start the Flask app and serve static content
# CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
