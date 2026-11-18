from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import yaml
import xml.etree.ElementTree as ET

app = Flask(__name__)

# PROBLEM: Multiple models loaded
pricing_model = joblib.load('../../ml-models/models/dynamic_pricing.pkl')
recommender = joblib.load('../../ml-models/models/recommender_v2.pkl')

@app.route('/api/pricing/calculate', methods=['POST'])
def calculate_price():
    data = request.json
    
    # PROBLEM: Explosive glue code - multiple conversions
    df = pd.DataFrame([data])
    csv_string = df.to_csv()
    import io
    csv_data = pd.read_csv(io.StringIO(csv_string))
    records = csv_data.to_dict('records')
    
    # XML conversion
    root = ET.Element("product")
    for key, value in records[0].items():
        child = ET.SubElement(root, key)
        child.text = str(value)
    xml_str = ET.tostring(root, encoding='unicode')
    
    # Use multiple models
    price = pricing_model.predict([list(data.values())])[0]
    rec_score = recommender.predict_proba([list(data.values())])[0][1]
    
    final_price = price * (1 + rec_score * 0.1)
    
    return jsonify({'price': float(final_price)})

if __name__ == '__main__':
    app.run(port=5007)
