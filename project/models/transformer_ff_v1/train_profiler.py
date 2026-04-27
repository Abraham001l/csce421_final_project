from transformers import AutoTokenizer
from transformer_ff import transformer_ff
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import sys
import time

# prevent the tokenizer from deadlocking the Dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset

def format_memory(bytes_val):
    return f"{bytes_val / (1024 ** 3):.2f} GB"

def main():
    print("=== PyTorch Profiler Initializing ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: CUDA is not available. Memory profiling will be skipped.")

    use_precomputed = True  # Keep this identical to your planned training run

    # ----- 1. Load Data (Subset to save time if needed, but we'll use full here) -----
    print("\nLoading data...")
    if use_precomputed:
        X_train = torch.load('../../data/mimic_data/train_sapbert_embeddings.pt')
        y_train = torch.load('../../data/mimic_data/train_sapbert_labels.pt')
    else:
        data = pd.read_csv('../../data/mimic_data/train_data.csv')
        X_train = data['TEXT'].tolist()
        y_train = data['ICD9_CODE'].astype(int).values

    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    train_dataset = mimic_dataset(X_train, y_train, tokenizer, precomputed=use_precomputed)
    
    batch_size = 32
    print(f"Dataset size: {len(train_dataset)} | Batch size: {batch_size}")
    total_batches = len(train_dataset) // batch_size

    # ----- PHASE 1: Dataloader Worker Tuning -----
    print("\n=== PHASE 1: Dataloader Worker Tuning ===")
    worker_counts = [0, 2, 4, 8, 12, 16]
    test_batches = 100 # How many batches to test per worker setting
    
    best_time = float('inf')
    best_workers = 0

    for workers in worker_counts:
        # Create a fresh dataloader for each test
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=workers, pin_memory=False)
        
        start_time = time.time()
        for i, (texts_embeddings, labels) in enumerate(loader):
            if i >= test_batches:
                break
            # We don't do anything here, just testing how fast the CPU feeds the loop
            pass
        
        elapsed = time.time() - start_time
        print(f"num_workers={workers:2d} | Time for {test_batches} batches: {elapsed:.2f} seconds")
        
        if elapsed < best_time:
            best_time = elapsed
            best_workers = workers

    print(f"-> Optimal num_workers seems to be: {best_workers}")

    # ----- PHASE 2: Memory Profiling -----
    print("\n=== PHASE 2: VRAM Memory Profiling ===")
    # Re-create dataloader with the optimal workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=best_workers, pin_memory=False)

    model = transformer_ff(
        dropout=0.2, 
        hidden1_size=256, 
        hidden2_size=128, 
        device=device,
        load_sapbert=(not use_precomputed)
    ).to(device)

    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Base VRAM (CUDA Context + Model Weights): {format_memory(torch.cuda.memory_allocated())}")

    model.train()
    profile_steps = 10
    
    print(f"Running {profile_steps} steps to measure peak VRAM...")
    for i, (texts_embeddings, labels) in enumerate(train_loader):
        if i >= profile_steps:
            break
            
        if (not use_precomputed):
            texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
        else:
            texts_embeddings = texts_embeddings.to(device)
            
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(texts_embeddings)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated()
        total_vram = torch.cuda.get_device_properties(device).total_memory
        print(f"Peak VRAM during training step: {format_memory(peak_vram)}")
        print(f"Total GPU VRAM Available: {format_memory(total_vram)}")
        print(f"VRAM Utilization: {(peak_vram/total_vram)*100:.1f}%")

    # ----- PHASE 3: Throughput & Time Estimation -----
    print("\n=== PHASE 3: Throughput & Time Estimation ===")
    throughput_steps = 100
    
    if throughput_steps > total_batches:
        throughput_steps = total_batches

    start_time = time.time()
    for i, (texts_embeddings, labels) in enumerate(train_loader):
        if i >= throughput_steps:
            break
            
        if (not use_precomputed):
            texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
        else:
            texts_embeddings = texts_embeddings.to(device)
            
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(texts_embeddings)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    elapsed = time.time() - start_time
    time_per_batch = elapsed / throughput_steps
    estimated_epoch_time = time_per_batch * total_batches

    print(f"Time for {throughput_steps} training steps: {elapsed:.2f} seconds")
    print(f"Average time per batch: {time_per_batch:.4f} seconds")
    print(f"Total batches per epoch: {total_batches}")
    print(f"-> Estimated time for 1 FULL epoch: {estimated_epoch_time / 60:.2f} minutes")
    print(f"-> Estimated time for 10 epochs: {(estimated_epoch_time * 10) / 3600:.2f} hours")
    
    print("\n=== Profiling Complete ===")

if __name__ == "__main__":
    main()