import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models.cnn import SimpleCifarCNN
from src.utils import get_loaders

def evaluate(model_path="outputs/best_model.pt", out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders()

    model = SimpleCifarCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds); all_targets.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)

if __name__ == "__main__":
    evaluate()
