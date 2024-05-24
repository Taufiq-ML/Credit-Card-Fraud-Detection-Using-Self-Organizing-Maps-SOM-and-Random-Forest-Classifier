


# Credit Card Fraud Detection using SOM and Random Forest

This project demonstrates a method to detect fraudulent credit card transactions using Self-Organizing Maps (SOM) for feature extraction and a Random Forest classifier for classification. The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle.

## Project Overview

Fraud detection is a critical task in the financial sector, where identifying fraudulent transactions can save significant amounts of money and enhance customer trust. This project leverages SOM for unsupervised feature extraction, followed by a Random Forest classifier to achieve high accuracy in detecting fraudulent transactions.

## Dataset

The dataset used for this project is from Kaggle, and it contains anonymized credit card transactions labeled as either fraudulent or genuine.

- [Credit Card Fraud Detection Dataset by MLG-ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Project Structure

- `creditcard.csv`: The dataset file.
- `train_and_save_models.py`: Script for training the models and saving them.
- `predict_new_transaction.py`: Script for loading the models and predicting new transactions.
- `README.md`: Project documentation.
- `potential_frauds.csv`: The output file containing potential fraudulent transactions identified by the model.

## Requirements

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `minisom`
- `joblib`

You can install these libraries using pip:

```sh
pip install pandas numpy scikit-learn minisom joblib
```

## Usage

### Training and Saving Models

1. **Clone the repository:**

```sh
git clone https://github.com/Taufiq-ML/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Download the dataset:**

   Place the `creditcard.csv` file in the root directory of the project. You can download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

3. **Run the training script:**

```sh
python train_and_save_models.py
```

   This script will train the models and save them as `scaler.pkl`, `som.pkl`, and `classifier.pkl`.

### Predicting New Transactions

1. **Run the prediction script:**

```sh
python predict_new_transaction.py
```

   This script will load the saved models and predict whether the provided transaction is fraudulent or not.

### Example Usage

Modify the `new_user_transaction` dictionary in `predict_new_transaction.py` with the actual transaction data you want to check. Here is an example:

```python
new_user_transaction = pd.DataFrame([{
    'Time': 100000, 'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536346, 'V4': 1.378155, 'V5': -0.338321,
    'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600,
    'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470400, 'V17': 0.207971,
    'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474,
    'V24': 0.066928, 'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62
}])
```

## Methodology

1. **Data Preprocessing:**
   - Load and standardize the dataset.
   
2. **Self-Organizing Maps (SOM):**
   - Train SOM to map high-dimensional data to a 2D grid.
   
3. **Feature Extraction:**
   - Use SOM to extract features by mapping each transaction to its corresponding winning neuron.
   
4. **Random Forest Classifier:**
   - Train a Random Forest classifier on the SOM features.
   - Evaluate the classifier on a test set.
   - Save the transactions predicted as fraudulent.

## Results

The combination of SOM for feature extraction and a Random Forest classifier achieves high accuracy in detecting fraudulent transactions. The results are summarized in the classification report and accuracy score output by the script.

runfile('C:/Users/UseR/Desktop/Fraud/train_and_save_models.py', wdir='C:/Users/UseR/Desktop/Fraud')
SOM training completed.
Accuracy: 1.00
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.79      0.36      0.49       136

    accuracy                           1.00     85443
   macro avg       0.89      0.68      0.75     85443
weighted avg       1.00      1.00      1.00     85443

Potential fraudulent transactions saved to 'potential_frauds.csv'.

runfile('C:/Users/UseR/Desktop/Fraud/predict_new_transaction.py', wdir='C:/Users/UseR/Desktop/Fraud')
The transaction is: Not Fraudulent

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is provided by [MLG-ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) on Kaggle.
- The SOM implementation is based on the `minisom` library.
```

