from torch.utils.data import Dataset, DataLoader
import torch

class mimic_dataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

        # printing shapes for debugging
        print(f"shape of label: {torch.tensor(self.labels[0], dtype=torch.float32).shape}")

    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.text[idx], torch.tensor(self.labels[idx], dtype=torch.float32)