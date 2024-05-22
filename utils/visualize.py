from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    train_loss = np.loadtxt("./results/FCN_DAE/train_loss.txt")
    val_loss = np.loadtxt("./results/FCN_DAE/val_loss.txt")

    train_epoch_loss = [np.mean(train_loss[i:i+36]) for i in range(0, len(train_loss), 36)]
    val_epoch_loss = [np.mean(val_loss[i:i+10]) for i in range(0, len(val_loss), 10)]

    ax.plot(train_epoch_loss, label="Train Loss")
    ax.plot(val_epoch_loss, label="Validation Loss")
    ax.set_title("FCN_DAE Loss, Learning Rate 0.0001")
    ax.legend()
    plt.tight_layout()
    Path("./results/FCN_DAE").mkdir(parents=True, exist_ok=True)
    plt.savefig("./results/FCN_DAE/loss.png")