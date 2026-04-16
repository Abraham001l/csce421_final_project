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
X_train = np.load('../helper_code/X_train.npy')
y_train = np.load('../helper_code/y_train.npy')
X_val = np.load('../helper_code/X_val.npy')
y_val = np.load('../helper_code/y_val.npy')

# ----- training logistic regression model -----
logistic_model = logistic_regression_model()
logistic_model.fit(X_train, y_train)

# ----- evaluating the model on the validation set -----
y_pred = logistic_model.predict(X_val)
print(classification_report(y_val, y_pred))

# ----- saving the trained model -----
logistic_model.save_model('logistic_model.joblib')