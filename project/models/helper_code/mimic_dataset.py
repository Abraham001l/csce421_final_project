from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

class mimic_dataset(Dataset):
    def __init__(self, text, labels, tokenizer, precomputed=False):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.precomputed = precomputed

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.precomputed:
            item = self.text[idx].clone().detach()
            return item, label
        
        text = self.text[idx]
        encoding = self.tokenizer(text, 
                                  padding='max_length', 
                                  truncation=True, 
                                  return_tensors='pt', 
                                  max_length=512)
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return item, label