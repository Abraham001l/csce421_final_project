import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import optuna
from optuna.pruners import HyperbandPruner
from tqdm import tqdm

# prevent the tokenizer from deadlocking the Dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from helper_code.mimic_dataset import mimic_dataset
from transformer_ff import transformer_ff

def objective(trial, train_loader, val_loader, device, use_precomputed=False):
    """
    Optuna objective function. Trains the model with a specific set of 
    hyperparameters and returns the validation metric to optimize.
    """
    # ----- define hyperparameter search space -----
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden1_size = trial.suggest_categorical("hidden1_size", [128, 256, 512])
    hidden2_size = trial.suggest_categorical("hidden2_size", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # ----- initialize model, loss, optimizer -----
    model = transformer_ff(
        dropout=dropout, 
        hidden1_size=hidden1_size, 
        hidden2_size=hidden2_size, 
        device=device,
        load_sapbert=(not use_precomputed)
    )
    model.to(device)
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    num_epochs = 10
    best_f1 = 0.0

    # ----- training & validation loop -----
    for epoch in range(num_epochs):
        # --- train ---
        model.train()

        train_bar = tqdm(train_loader, desc=f"Trial {trial.number} - Epoch {epoch}", leave=False)

        for texts_embeddings, labels in train_bar:
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

            train_bar.set_postfix(loss=loss.item())

        # --- evaluate ---
        model.eval()
        total_val_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        val_bar = tqdm(val_loader, desc=f"Trial {trial.number} - Validation", leave=False)

        with torch.no_grad():
            for texts_embeddings, labels in val_loader:
                if (not use_precomputed):
                    texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
                else:
                    texts_embeddings = texts_embeddings.to(device)
                labels = labels.to(device).view(-1, 1)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(texts_embeddings)
                    loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                predictions = (outputs > 0).int().view(-1)
                targets = (labels > 0).int().view(-1)

                tp += ((predictions == 1) & (targets == 1)).sum().item()
                fp += ((predictions == 1) & (targets == 0)).sum().item()
                tn += ((predictions == 0) & (targets == 0)).sum().item()
                fn += ((predictions == 0) & (targets == 1)).sum().item()

                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        best_f1 = max(best_f1, f1)

        # ----- hyperband pruning logic -----
        # report the metric you want to base pruning on (F1 score in this case)
        trial.report(f1, epoch)

        # handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- load and subsample data (10%) -----
    use_precomputed = True  # Set to False to compute embeddings on the fly (not recommended for large datasets)
    X, y = None, None
    if use_precomputed:
        print("Loading precomputer embeddings...")
        X = torch.load('../../data/mimic_data/train_sapbert_embeddings.pt')
        y = torch.load('../../data/mimic_data/train_sapbert_labels.pt')
        # sampling 10% of the dataset to speed up tuning
        indices = torch.randperm(len(X))[:len(X) // 10]
        X = X[indices]
        y = y[indices]
    else:
        data = pd.read_csv('../../data/mimic_data/train_data.csv')
        # sample only 10% of the dataset to speed up tuning
        data = data.sample(frac=0.1, random_state=42)
        X = data['TEXT'].tolist()
        y = data['ICD9_CODE'].astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- create datasets and dataloaders once -----
    # Moving this outside the objective function prevents tokenizing the dataset 
    # from scratch on every single hyperparameter trial.
    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    train_dataset = mimic_dataset(X_train, y_train, tokenizer, precomputed=use_precomputed)
    val_dataset = mimic_dataset(X_val, y_val, tokenizer, precomputed=use_precomputed)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    # ----- setup optuna study with hyperband -----
    # min_resource: epoch at which to start pruning. max_resource: total epochs.
    pruner = HyperbandPruner(min_resource=2, max_resource=10, reduction_factor=3)
    
    # We want to MAXIMIZE the F1 score. If you want to optimize for validation loss, 
    # change this to "minimize" and return avg_val_loss in the objective.
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    # setting optuna logger
    optuna.logging.get_logger("optuna").setLevel(optuna.logging.INFO)

    print("Starting Hyperband Tuning...")
    # Wrap the objective in a lambda to pass in our pre-loaded data
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device, use_precomputed), n_trials=250)

    # ----- output results -----
    print("\n--- Hyperband Tuning Completed ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (F1 Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()