from logistic_regression import logistic_regression_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import classification_report


# ----- setting up sapBERT tokenizer and model -----
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def get_sapbert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# ----- loading and preparing data -----
data = pd.read_csv('../../data/mimic_data/train_data.csv')
X = get_sapbert_embeddings(data['TEXT'].tolist())
y = data['ICD9_CODE'].notnull().astype(int).values

# ----- splitting data into training and testing sets -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- loading the trained logistic regression model -----
logistic_model = logistic_regression_model()
logistic_model.load_model('logistic_model.joblib')

# ----- making predictions on the test set -----
y_pred = logistic_model.predict(X_test)

# ----- evaluating the model -----
print(classification_report(y_test, y_pred))