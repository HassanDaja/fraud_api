from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the trained pipeline
model = joblib.load('fraud_detection_pipeline_cpu.pkl')

@app.route('/')
def home():
    return "Hello from Flask!"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract features from the request
        features = {
            'step': [data['step']],
            'type': [data['type']],
            'amount': [data['amount']],
            'oldbalanceOrg': [data['oldbalanceOrg']],
            'newbalanceOrig': [data['newbalanceOrig']],
            'oldbalanceDest': [data['oldbalanceDest']],
            'newbalanceDest': [data['newbalanceDest']]
        }
        input_df = pd.DataFrame(features)
        # Make prediction
        prediction = model.predict(input_df)
        result = 'Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
