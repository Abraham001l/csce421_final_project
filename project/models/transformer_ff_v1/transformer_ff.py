import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class transformer_ff(nn.Module):
    def __init__(self, dropout=0.2, hidden1_size=256, hidden2_size=128, device=None, load_sapbert=True):
        super(transformer_ff, self).__init__()

        self.device = device
        self.load_sapbert = load_sapbert
        
        if self.load_sapbert:
            self.sapbert = AutoModel.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
        else:
            self.sapbert = None

        self.ff = nn.Sequential(
            nn.Linear(768, hidden1_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1_size, hidden2_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2_size, 1),
        )
    
    def forward(self, x):
        if not self.load_sapbert:
            cls_embeddings = x
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.sapbert(**x)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return self.ff(cls_embeddings).float()