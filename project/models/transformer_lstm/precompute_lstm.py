import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset

# -----------------------------------------------------------------------
# STORAGE NOTE:
#   Unlike the DNN precompute (which only saved the [CLS] token at 768-dim),
#   the LSTM needs the full token sequence: shape [N, seq_len, 768].
#
#   Rough size estimates (float16):
#     seq_len=128 → ~9.8 GB per 50k samples
#     seq_len=256 → ~19.7 GB per 50k samples
#     seq_len=512 → ~39.3 GB per 50k samples
#
#   Embeddings are stored in float16 to halve disk/RAM usage.
#   They are cast back to float32 automatically inside the model forward pass.
# -----------------------------------------------------------------------

def precompute_lstm_embeddings(split: str = "train"):
    """
    Args:
        split: one of "train" or "test" — controls which CSV is read
               and which output files are written.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Load raw data
    # ------------------------------------------------------------------ #
    csv_path = f'../../data/mimic_data/{split}_data.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    data = pd.read_csv(csv_path)
    X = data['TEXT'].tolist()
    y = data['ICD9_CODE'].astype(int).values
    print(f"Loaded {len(X)} samples from {csv_path}")

    # ------------------------------------------------------------------ #
    # 2. Setup tokenizer and SapBERT
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(
        "../helper_code/sapBERT_local_save", local_files_only=True
    )
    sapbert = AutoModel.from_pretrained(
        "../helper_code/sapBERT_local_save", local_files_only=True
    )
    sapbert.to(device)
    sapbert.eval()

    # ------------------------------------------------------------------ #
    # 3. DataLoader  (precomputed=False → tokenise raw text on the fly)
    # ------------------------------------------------------------------ #
    dataset = mimic_dataset(X, y, tokenizer, precomputed=False)
    # Smaller batch size than DNN precompute because we keep the full
    # sequence (seq_len × 768) in GPU memory before moving to CPU.
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    all_embeddings   = []   # will be List[Tensor shape (B, seq_len, 768)]
    all_masks        = []   # will be List[Tensor shape (B, seq_len)]
    all_labels       = []

    # ------------------------------------------------------------------ #
    # 4. Forward pass — keep ALL token embeddings, not just [CLS]
    # ------------------------------------------------------------------ #
    print(f"Pre-computing SapBERT sequence embeddings for '{split}' split...")
    with torch.no_grad():
        for texts_dict, labels in tqdm(dataloader, leave=False):
            attention_mask = texts_dict['attention_mask']   # (B, seq_len)

            texts_dict = {k: v.to(device) for k, v in texts_dict.items()}

            with torch.amp.autocast(device_type=device.type):
                outputs = sapbert(**texts_dict)

            # Full sequence embeddings: (B, seq_len, 768)
            seq_embeddings = outputs.last_hidden_state

            # Store in float16 to halve memory/disk usage.
            # Cast back to float32 inside the model's forward().
            all_embeddings.append(seq_embeddings.cpu().to(torch.float16))
            all_masks.append(attention_mask.cpu())          # already int/bool
            all_labels.append(labels.cpu())

    # ------------------------------------------------------------------ #
    # 5. Concatenate and save
    # ------------------------------------------------------------------ #
    final_embeddings = torch.cat(all_embeddings, dim=0)   # (N, seq_len, 768)
    final_masks      = torch.cat(all_masks,      dim=0)   # (N, seq_len)
    final_labels     = torch.cat(all_labels,     dim=0)   # (N,)

    save_dir = '../../data/mimic_data/'
    os.makedirs(save_dir, exist_ok=True)

    torch.save(final_embeddings, os.path.join(save_dir, f'{split}_sapbert_lstm_embeddings.pt'))
    torch.save(final_masks,      os.path.join(save_dir, f'{split}_sapbert_lstm_masks.pt'))
    torch.save(final_labels,     os.path.join(save_dir, f'{split}_sapbert_lstm_labels.pt'))

    emb_gb = final_embeddings.element_size() * final_embeddings.nelement() / 1e9
    print(
        f"\nSaved {final_embeddings.shape[0]} samples | "
        f"seq_len={final_embeddings.shape[1]} | "
        f"hidden={final_embeddings.shape[2]} | "
        f"disk size ≈ {emb_gb:.2f} GB  (float16)"
    )


if __name__ == "__main__":
    # Precompute both splits.  Comment out whichever you don't need.
    precompute_lstm_embeddings(split="train")
    precompute_lstm_embeddings(split="test")
