from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/crop_production_prediction_model.joblib")

# Function to encode categorical variables (if necessary)
def preprocess_input(data):
    # Convert categorical features if required
    df = pd.DataFrame([data], columns=["year", "district", "crop", "area", "yield"])
    
    # Encoding categorical variables (Assume district & crop need encoding)
    df["district"] = df["district"].astype("category").cat.codes
    df["crop"] = df["crop"].astype("category").cat.codes
    
    return df.values.reshape(1, -1)  # Convert to NumPy array (2D)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features and preprocess
        features = preprocess_input(data["features"])

        # Make prediction
        prediction = model.predict(features)

        # Send response
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
