from pathlib import Path
from tqdm import tqdm, trange

import numpy as np
import os
import torch
import wfdb

from models import FCN_DAE

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

torch.random.manual_seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# check and select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise(signal, noise_type, snr, train=True):
    assert noise_type in ["bw", "em", "ma"], "Noise type must be one of ['bw', 'em', 'ma']"
    st = np.random.randint(0, bw.shape[0]-signal.shape[0]) if train else 0
    if noise_type == "bw":
        noise = bw[st:st+signal.shape[0], 0]
    elif noise_type == "em":
        noise = em[st:st+signal.shape[0], 0]
    elif noise_type == "ma":
        noise = ma[st:st+signal.shape[0], 0]
    
    # mix signal and noise with SNR
    rms_clean = np.sqrt(np.mean(signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    scale = rms_clean / (10**(snr / 20) * rms_noise)
    noisy = signal + noise * scale

    return noisy

if __name__ == "__main__":
    # TODO: use generator to load data and make it more efficient

    # preserve 118 and 119 for testing as they are in NSTDB
    mitdb_subjects = [s for s in wfdb.io.get_record_list("mitdb") if s not in ["118", "119"]]
    val_subjects = np.random.choice(mitdb_subjects, 10) # 10 subjects for validation
    train_subjects = [s for s in mitdb_subjects if s not in val_subjects]

    bw, _ = wfdb.rdsamp("./data/physionet.org/files/nstdb/1.0.0/bw", channels=[0])
    em, _ = wfdb.rdsamp("./data/physionet.org/files/nstdb/1.0.0/em", channels=[0])
    ma, _ = wfdb.rdsamp("./data/physionet.org/files/nstdb/1.0.0/ma", channels=[0])
    
    # hyperparameters
    lr = 0.0001
    epochs = 30

    # load model
    model = FCN_DAE().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_loss = -np.inf
    train_loss = []
    val_loss = []
    for epoch in trange(epochs, desc="FCN_DAE Training"):
        # train
        model.train()
        for t, train_subject in tqdm(enumerate(train_subjects), desc=f"Train {epoch+1} Epoch", leave=False):
            random_start = np.random.randint(0, 650000 % 1024)  # 650000 is the maximum length of the record, 1024 is the window size
            signal, _ = wfdb.rdsamp(f"./data/physionet.org/files/mitdb/1.0.0/{train_subject}", channels=[0])   # only use first channel
            
            train_step_loss = 0
            count = 0
            for i in tqdm(range(random_start, 650000, 1024), desc="Step", leave=False):
                clean_signal = signal[i:i+1024][:, 0]
                if clean_signal.shape[0] != 1024:
                    continue
                noisy_signal = add_noise(clean_signal, "bw", np.random.randint(-6, 24))

                clean_signal = torch.from_numpy(clean_signal.reshape(1, 1, 1024)).to(DEVICE)
                noisy_signal = torch.from_numpy(noisy_signal.reshape(1, 1, 1024)).to(DEVICE)
                
                count += 1

                optimizer.zero_grad()
                output, _ = model(noisy_signal.float())
                loss = criterion(output, clean_signal.float())
                train_step_loss += loss.item()
                loss.backward()
                optimizer.step()
            average_step_loss = train_step_loss / count
            train_loss.append(average_step_loss)
        scheduler.step()

        # validation
        model.eval()
        current_val_loss = 0
        for val_subject in tqdm(val_subjects, desc=f"Validate {epoch+1} Epoch", leave=False):
            signal, _ = wfdb.rdsamp(f"./data/physionet.org/files/mitdb/1.0.0/{val_subject}", channels=[0])   # only use first channel
            
            val_step_loss = 0
            count = 0
            for j in tqdm(range(0, 650000, 1024), desc="Step", leave=False):
                clean_signal = signal[j:j+1024][:, 0]
                if clean_signal.shape[0] != 1024:
                    continue
                noisy_signal = add_noise(clean_signal, "bw", -6, train=False)

                clean_signal = torch.from_numpy(clean_signal.reshape(1, 1, 1024)).to(DEVICE)
                noisy_signal = torch.from_numpy(noisy_signal.reshape(1, 1, 1024)).to(DEVICE)
                
                count += 1

                output, _ = model(noisy_signal.float())
                loss = criterion(output, clean_signal.float())
                val_step_loss += loss.item()
            average_val_loss = val_step_loss / count
            val_loss.append(average_val_loss)
            current_val_loss += average_val_loss
        
        if current_val_loss < best_loss:
            best_loss = average_val_loss
            Path("./weights").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"./weights/FCN_DAE_{str(lr).split('.')[-1]}.pth")

            