# Shared pipeline used by multiple services
import pandas as pd
import numpy as np

def calculate_customer_score(customer_id, data_source='database'):
    # Used by customer-service, recommendation-service, fraud-detection
    return 0.75

def build_feature_vector(customer):
    # Different formats for different models
    fraud_features = [customer.get('age', 0), customer.get('total_spent', 0)]
    rec_features = [customer.get('total_spent', 0), customer.get('age', 0)]
    
    return {
        'fraud_features': fraud_features,
        'rec_features': rec_features
    }
