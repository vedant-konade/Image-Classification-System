import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.cnn import SimpleCifarCNN
from src.utils import get_loaders, accuracy_from_logits

def train_model(epochs=30, lr=1e-3, patience=5, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCifarCNN(num_classes=10, p_drop=0.3).to(device)
    train_loader, val_loader, _ = get_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    best_path = os.path.join(out_dir, "best_model.pt")
    no_improve = 0

    tr_losses, val_losses = [], []
    tr_accs, val_accs = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            b = x.size(0)
            running_loss += loss.item() * b
            running_acc  += accuracy_from_logits(logits, y) * b
            n            += b

        tr_loss = running_loss / n
        tr_acc  = running_acc / n

        # validation
        model.eval()
        v_loss, v_acc, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                b = x.size(0)
                v_loss += loss.item() * b
                v_acc  += accuracy_from_logits(logits, y) * b
                vn     += b
        val_loss = v_loss / vn
        val_acc  = v_acc / vn

        tr_losses.append(tr_loss); val_losses.append(val_loss)
        tr_accs.append(tr_acc);    val_accs.append(val_acc)

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    # Save curves
    plt.figure()
    plt.plot(tr_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, "curves.png"), dpi=150)

    plt.figure()
    plt.plot(tr_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(out_dir, "curves_loss.png"), dpi=150)

    print(f"Best model saved to: {best_path}")

if __name__ == "__main__":
    train_model()
