from transformer_ff import transformer_ff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from helper_code.mimic_dataset import mimic_dataset

# ----- load the training and validation data -----
data = pd.read_csv('../../data/mimic_data/train_data.csv')
X = data['TEXT'].tolist()
y = data['ICD9_CODE'].astype(int).values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- create datasets and dataloaders -----
train_dataset = mimic_dataset(X_train, y_train)
val_dataset = mimic_dataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----- initialize the model, loss function, and optimizer -----
model = transformer_ff()
device = model.device
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ----- training loop -----
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for texts, labels in train_loader:
#         labels = labels.to(device)

#         # forward pass
        
