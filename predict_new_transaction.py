import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load necessary models and scalers
scaler = joblib.load('scaler.pkl')
som = joblib.load('som.pkl')
clf = joblib.load('classifier.pkl')

# Function to check if a specific user's transaction is fraudulent
def check_fraudulent_transaction(user_transaction):
    # Ensure user_transaction is a DataFrame with the same structure as the features in the original dataset
    user_transaction_scaled = scaler.transform(user_transaction)
    
    # Map the user's transaction to SOM features
    winning_node = som.winner(user_transaction_scaled[0])
    som_feature = np.array([winning_node])
    
    # Predict using the Random Forest classifier
    prediction = clf.predict(som_feature)
    
    return 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'

# Example usage
# Assume new_user_transaction is a DataFrame containing the new user's transaction data
new_user_transaction = pd.DataFrame([{
    'Time': 100000, 'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536346, 'V4': 1.378155, 'V5': -0.338321,
    'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600,
    'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470400, 'V17': 0.207971,
    'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474,
    'V24': 0.066928, 'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62
}])

result = check_fraudulent_transaction(new_user_transaction)
print(f"The transaction is: {result}")

