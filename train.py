import os

from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import FCN_DAE
from datasets.noisy_ecg import NOISY_ECG

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

torch.random.manual_seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# check and select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # hyperparameters
    lr = 0.0001
    batch_size = 64
    epochs = 64

    # load model
    model = FCN_DAE().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # load dataset
    train_dataloader = DataLoader(
        NOISY_ECG(root="./data/physionet.org/files/", channel=[0], window_size=1024, train=True),
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        NOISY_ECG(root="./data/physionet.org/files/", channel=[0], window_size=1024, train=False),
        batch_size=batch_size,
    )
    max_iter_per_epoch = 1024

    best_loss = np.inf
    average_train_loss = []
    average_val_loss = []
    for epoch in trange(epochs, desc="FCN_DAE Training"):
        # train
        model.train()
        train_loss = 0
        for i, (noisy_signal, clean_signal) in tqdm(enumerate(train_dataloader), desc=f"Train {epoch+1} Epoch", total=max_iter_per_epoch, leave=False):
            optimizer.zero_grad()
            output, _ = model(noisy_signal.to(DEVICE))
            loss = criterion(output, clean_signal.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i >= max_iter_per_epoch:
                break
        average_train_loss.append(train_loss / (i+1))
        print(f"Epoch {epoch+1} - Train Loss: {train_loss / (i+1)}")
        scheduler.step()

        # validation
        model.eval()
        val_loss = 0
        for i, (noisy_signal, clean_signal) in tqdm(enumerate(val_dataloader), desc=f"Validate {epoch+1} Epoch", total=max_iter_per_epoch, leave=False):
            output, _ = model(noisy_signal.to(DEVICE))
            loss = criterion(output, clean_signal.to(DEVICE))
            val_loss += loss.item()
            if i >= max_iter_per_epoch:
                break
        average_val_loss.append(val_loss / (i+1))
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss / (i+1)}")
        # save best model
        if float(val_loss / (i+1)) < best_loss:
            best_loss = float(val_loss / (i+1))
            Path("./weights/FCN_DAE").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"./weights/FCN_DAE/lr{str(lr).split('.')[-1]}_b{batch_size}_e{epochs}.pth")

    # save loss
    Path(f"./results/FCN_DAE/lr{str(lr).split('.')[-1]}_b{batch_size}_e{epochs}").mkdir(parents=True, exist_ok=True)
    np.savetxt(f"./results/FCN_DAE/lr{str(lr).split('.')[-1]}_b{batch_size}_e{epochs}/train_loss.txt", np.array(train_loss), fmt="%.4f")
    np.savetxt(f"./results/FCN_DAE/lr{str(lr).split('.')[-1]}_b{batch_size}_e{epochs}/val_loss.txt", np.array(val_loss), fmt="%.4f")

            