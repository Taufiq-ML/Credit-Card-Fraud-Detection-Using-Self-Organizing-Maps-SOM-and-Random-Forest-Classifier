import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from minisom import MiniSom
import joblib

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Separate features and labels
features = data.drop(columns=['Class'])
labels = data['Class']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize and train SOM
som = MiniSom(x=15, y=15, input_len=features_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(features_scaled)
som.train_random(features_scaled, 100)
print("SOM training completed.")

# Map each data point to its corresponding winning neuron on the SOM
def map_to_som(som, data):
    mapped_data = []
    for sample in data:
        winning_node = som.winner(sample)
        mapped_data.append(winning_node)
    return np.array(mapped_data)

# Map the training data to SOM features
som_features = map_to_som(som, features_scaled)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(som_features, labels, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(som, 'som.pkl')
joblib.dump(clf, 'classifier.pkl')

# Save the potential fraudulent transactions based on classifier prediction
potential_frauds = data.iloc[y_test.index[y_pred == 1]]
potential_frauds.to_csv('potential_frauds.csv', index=False)
print("Potential fraudulent transactions saved to 'potential_frauds.csv'.")


