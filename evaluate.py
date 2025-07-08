import os
import torch
import numpy as np
from autoencoder import MaskedDenoisingAutoencoder
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = "cluster_models"
DATA_DIR = "preprocessed"
THRESHOLD_PERCENTILE = 95

def load_data(dir_path):
    """Load data and mask arrays from directory"""
    data = np.load(os.path.join(dir_path, "data.npy"))
    mask = np.load(os.path.join(dir_path, "mask.npy"))
    return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def compute_threshold(model, data, mask):
    """Compute anomaly threshold using 95th percentile of MSE"""
    model.eval()
    with torch.no_grad():
        recon = model(data, mask)
        mse = ((recon - data) ** 2 * mask).sum(dim=1) / mask.sum(dim=1)
    return np.percentile(mse.numpy(), THRESHOLD_PERCENTILE)

def evaluate_file(model, threshold, data, mask):
    """Evaluate model on data and return binary predictions"""
    model.eval()
    with torch.no_grad():
        recon = model(data, mask)
        mse = ((recon - data) ** 2 * mask).sum(dim=1) / mask.sum(dim=1)
        preds = (mse > threshold).int().numpy()
    return preds


def run_evaluation():
    """Main evaluation loop for all protocols"""
    for model_file in os.listdir(MODELS_DIR):
        if not model_file.endswith(".pth"):
            continue

        protocol = model_file.replace("cluster_", "").replace(".pth", "")
        model_path = os.path.join(MODELS_DIR, model_file)
        print(f"\nEvaluating for protocol: {protocol}")

        # Find validation file for threshold computation
        val_dir = None
        for name in os.listdir(DATA_DIR):
            if (protocol in name.lower() and "client" not in name.lower()
                    and "normal" in name.lower() and "test" not in name.lower()):
                val_dir = os.path.join(DATA_DIR, name)
                break

        if not val_dir:
            print(f"No validation file for threshold found for protocol {protocol}, skipping.")
            continue

        # Load validation data and compute threshold
        x_val, m_val = load_data(val_dir)
        input_dim = x_val.shape[1]

        model = MaskedDenoisingAutoencoder(input_dim)
        model.load_state_dict(torch.load(model_path))
        threshold = compute_threshold(model, x_val, m_val)
        print(f"Threshold (95th percentile MSE): {threshold:.4f}")

        # Evaluate on all test files
        all_preds = []
        all_labels = []
        for name in os.listdir(DATA_DIR):
            if (protocol in name.lower()
                and "client" not in name.lower()
                and "test" not in name.lower()
                and not name.endswith(".csv")):

                filepath = os.path.join(DATA_DIR, name)
                x, m = load_data(filepath)
                preds = evaluate_file(model, threshold, x, m)

                # Assign labels based on filename
                label = 0 if "normal" in name.lower() else 1
                labels = np.full_like(preds, label)

                all_preds.append(preds)
                all_labels.append(labels)

                print(f"{name}: Predicted {sum(preds)} anomalies out of {len(preds)}")

        # Generate classification report and confusion matrix
        if all_preds:
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            print(f"\nClassification Report for {protocol}:")
            print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    run_evaluation()
