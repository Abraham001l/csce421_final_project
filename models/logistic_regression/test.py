from logistic_regression import logistic_regression_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import classification_report

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- load the training and validation data -----
X_test = np.load('../helper_code/X_test.npy')
y_test = np.load('../helper_code/y_test.npy')

# ----- loading the trained logistic regression model -----
logistic_model = logistic_regression_model()
logistic_model.load_model('logistic_model.joblib')

# ----- making predictions on the test set -----
y_pred = logistic_model.predict(X_test)

# ----- evaluating the model -----
print(classification_report(y_test, y_pred))