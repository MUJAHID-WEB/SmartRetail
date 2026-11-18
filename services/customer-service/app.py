from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# PROBLEM: Direct model loading
fraud_model = joblib.load('../../ml-models/models/fraud_classifier.pkl')

@app.route('/api/customers/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    customer_data = {'id': customer_id, 'name': 'Test User', 'age': 30, 'total_spent': 1000}
    
    # PROBLEM: Direct prediction with transformations
    features = [customer_data['age'], customer_data['total_spent']]
    df = pd.DataFrame([features])
    fraud_score = fraud_model.predict_proba(df)[0][1]
    
    customer_data['fraud_risk'] = float(fraud_score)
    return jsonify(customer_data)

if __name__ == '__main__':
    app.run(port=5000)
