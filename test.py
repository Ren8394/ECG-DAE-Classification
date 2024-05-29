from pathlib import Path
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import wfdb

from models import FCN_DAE, BLSTM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

torch.random.manual_seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# check and select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    nstdb_subjects = [s for s in wfdb.io.get_record_list("nstdb") if s not in ["bw", "em", "ma"]]
    clean_signal_118, _ = wfdb.rdsamp(f"./data/physionet.org/files/mitdb/1.0.0/118", channels=[0])   # only use first channel for 118
    clean_signal_119, _ = wfdb.rdsamp(f"./data/physionet.org/files/mitdb/1.0.0/119", channels=[0])   # only use first channel for 119

    model = FCN_DAE().to(DEVICE)    # FCN_DAE
    # model = BLSTM().to(DEVICE)      # BLSTM
    model.load_state_dict(torch.load("./weights/FCN_DAE_lr0001.pth", map_location=DEVICE))

    # test model and visualize results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    mse_list = []
    for subject in nstdb_subjects:
        signal, _ = wfdb.rdsamp(f"./data/physionet.org/files/nstdb/1.0.0/{subject}", channels=[0])
        target_signal = clean_signal_118 if "118" in subject else clean_signal_119

        for j in tqdm(range(0, 650000, 1024), desc="Step", leave=False):
            noisy_signal = signal[j:j+1024][:, 0]
            clean_signal = target_signal[j:j+1024][:, 0]

            if clean_signal.shape[0] != 1024 or noisy_signal.shape[0] != 1024:
                    continue

            noisy_signal = torch.from_numpy(noisy_signal.reshape(1, 1, 1024)).to(DEVICE)
            clean_signal = torch.from_numpy(clean_signal.reshape(1, 1, 1024)).to(DEVICE)

            denoised_signal, _ = model(noisy_signal.float())
            mse = F.mse_loss(denoised_signal, clean_signal.float())
            mse_list.append(mse.item())

            clean_signal = clean_signal.detach().cpu().numpy().reshape(1024)
            noisy_signal = noisy_signal.detach().cpu().numpy().reshape(1024)
            denoised_signal = denoised_signal.detach().cpu().numpy().reshape(1024)
            ax.clear()
            ax.plot(clean_signal, label="Clean Signal")
            ax.plot(noisy_signal, label="Noisy Signal")
            ax.plot(denoised_signal, label="Denoised Signal")
            ax.set_title(f"{subject} - MSE: {mse.item()}")
            ax.legend()
            plt.tight_layout()
            Path(f"./results/{subject}").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"./results/{subject}/Step_{j}.png")

    print(f"Mean MSE: {np.mean(mse_list)}")