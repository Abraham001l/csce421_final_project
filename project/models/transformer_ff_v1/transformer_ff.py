import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class transformer_ff(nn.Module):
    def __init__(self, dropout=0.1, hidden1_size=256):
        super(transformer_ff, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
        self.sapbert = AutoModel.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)

        self.ff = nn.Sequential(
            nn.Linear(768, hidden1_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, 1),
        )

        self.to(self.device)
    
    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.sapbert(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.ff(cls_embeddings)