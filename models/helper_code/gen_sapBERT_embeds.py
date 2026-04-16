import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- setting up sapBERT tokenizer and model -----
local_model_path = "../helper_code/sapBERT_local_save"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(device)

def get_sapbert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# ----- loading and preparing data -----
data = pd.read_csv('../../data/mimic_data/train_data.csv')
X = get_sapbert_embeddings(data['TEXT'].tolist())
y = data['ICD9_CODE'].notnull().astype(int).values

# ----- splitting data into training and validation sets -----
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- saving the training and validation data for later use -----
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

# ----- loading and preparing test data -----
test_data = pd.read_csv('../../data/mimic_data/test_data.csv')
X_test = get_sapbert_embeddings(test_data['TEXT'].tolist())
y_test = test_data['ICD9_CODE'].notnull().astype(int).values

# ----- saving the test data for later use -----
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)