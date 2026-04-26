import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from mimic_dataset import mimic_dataset

def precompute_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load your raw data
    data = pd.read_csv('../../data/mimic_data/test_data.csv')
    
    X = data['TEXT'].tolist()
    y = data['ICD9_CODE'].astype(int).values

    # 2. Setup Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    sapbert = AutoModel.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    sapbert.to(device)
    sapbert.eval()

    # 3. Create a Dataloader (precomputed=False to tokenize raw text)
    dataset = mimic_dataset(X, y, tokenizer, precomputed=False)
    # Batch size can be larger since we aren't storing gradients
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    all_embeddings = []
    all_labels = []

    print("Pre-computing sapBERT embeddings...")
    with torch.no_grad():
        for texts_dict, labels in tqdm(dataloader, leave=False):
            # Move inputs to GPU
            texts_dict = {key: val.to(device) for key, val in texts_dict.items()}
            
            with torch.amp.autocast(device_type=device.type):
                outputs = sapbert(**texts_dict)
                
            # Extract [CLS] token embeddings (shape: [batch_size, 768])
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Move back to CPU to prevent GPU Out-of-Memory errors
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.append(labels.cpu())

    # Concatenate lists of tensors into single massive tensors
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # 4. Save to disk natively
    save_dir = '../../data/mimic_data/'
    torch.save(final_embeddings, os.path.join(save_dir, 'test_sapbert_embeddings.pt'))
    torch.save(final_labels, os.path.join(save_dir, 'test_sapbert_labels.pt'))
    
    print(f"Saved {final_embeddings.shape[0]} embeddings of size {final_embeddings.shape[1]}!")

if __name__ == "__main__":
    precompute_embeddings()