import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# prevent the tokenizer from deadlocking the Dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset
from transformer_ff import transformer_ff

def main():
    # ----- setting device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- configuration -----
    # Update this to the specific epoch you want to test (e.g., your best epoch)
    model_path = '../../results/transformer_ff_v1/model_epoch_10.pth'
    test_data_path = '../../data/mimic_data/test_data.csv' 
    results_dir = '../../results/transformer_ff_v1'

    # Ensure results directory exists for the plot
    os.makedirs(results_dir, exist_ok=True)

    # ----- load the testing data -----
    if not os.path.exists(test_data_path):
        print(f"Error: Could not find test data at {test_data_path}")
        return
        
    data = pd.read_csv(test_data_path)
    X_test = data['TEXT'].tolist()
    y_test = data['ICD9_CODE'].astype(int).values

    # ----- create datasets and dataloaders -----
    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    test_dataset = mimic_dataset(X_test, y_test, tokenizer)
    
    # shuffle=false is standard for testing
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    # ----- initialize the model, loss, and load weights -----
    model = transformer_ff(device=device)
    
    # Load the trained weights
    if not os.path.exists(model_path):
         print(f"Error: Could not find model weights at {model_path}")
         return
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    # ----- testing loop -----
    model.eval()
    total_test_loss = 0
    
    all_targets = []
    all_predictions = []
    all_outputs = [] # Raw outputs for AUC calculation

    # tqdm for progress bar
    test_bar = tqdm(test_loader, desc="[Testing]")

    with torch.no_grad():
        for texts_embeddings, labels in test_bar:
            texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
            labels = labels.to(device).view(-1, 1)
            
            outputs = model(texts_embeddings)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            # getting prediction and target
            predictions = (outputs > 0).int().view(-1)
            targets = labels.int().view(-1)

            # Store for sklearn metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_outputs.extend(outputs.view(-1).cpu().numpy())

            test_bar.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / len(test_loader)

    # ----- print final metrics -----
    print("\n" + "="*50)
    print("TESTING RESULTS")
    print("="*50)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print("-" * 50)
    
    print("Classification Report:")
    # zero_division=0 prevents warnings if a class is entirely missing
    print(classification_report(all_targets, all_predictions, zero_division=0))
    print("-" * 50)
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))
    
    try:
        # calculate ROC-AUC score 
        # note: We pass raw logit outputs here as sklearn handles the thresholding/ranking internally
        roc_auc = roc_auc_score(all_targets, all_outputs)
        print("-" * 50)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # ----- Plotting the ROC Curve -----
        fpr, tpr, thresholds = roc_curve(all_targets, all_outputs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal random guess line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plot_path = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"\nROC curve saved successfully to {plot_path}")

    except ValueError:
        # Fails gracefully if there's only one class present in the test subset
        print("-" * 50)
        print("ROC-AUC Score: N/A (Only one class present in testing data)")

if __name__ == "__main__":
    # avoiding child process spawning issues on Windows
    main()