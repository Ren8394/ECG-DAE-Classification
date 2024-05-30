from pathlib import Path
import os

from tqdm import tqdm, trange
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import matplotlib.pyplot as plt
import numpy as np
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

    model_name = "FCN_DAE"
    assert model_name in ["FCN_DAE", "BLSTM"], "Model name must be 'FCN_DAE' or 'BLSTM'"
    if model_name == "FCN_DAE":
        model = FCN_DAE(use_bn=False).to(DEVICE)    # FCN_DAE
        ckpt_path = "./weights/FCN_DAE/lr001_b64_e64.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
    elif model_name == "BLSTM":
        model = BLSTM().to(DEVICE)
        ckpt_path = "./weights/BLSTM/lr001_b64_e64.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)

    # test model and visualize results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    mse_list = []
    for subject in nstdb_subjects:
        signal, _ = wfdb.rdsamp(f"./data/physionet.org/files/nstdb/1.0.0/{subject}", channels=[0])
        target_signal = clean_signal_118 if "118" in subject else clean_signal_119

        for j in tqdm(range(0, 650000, 1024), desc=f"Step", leave=False):
            noisy_signal = signal[j:j+1024][:, 0]
            clean_signal = target_signal[j:j+1024][:, 0]

            if clean_signal.shape[0] != 1024 or noisy_signal.shape[0] != 1024:
                    continue
            
            noisy_signal = scaler.fit_transform(noisy_signal.reshape(-1, 1)).reshape(-1)
            clean_signal = scaler.fit_transform(clean_signal.reshape(-1, 1)).reshape(-1)

            clean_signal = torch.from_numpy(clean_signal.reshape(1, 1024)).float().to(DEVICE)
            noisy_signal = torch.from_numpy(noisy_signal.reshape(1, 1024)).float().to(DEVICE)

            denoised_signal, _ = model(noisy_signal.float())
            denoised_signal = denoised_signal.squeeze(0)
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
            Path(f"./results/{model_name}/{Path(ckpt_path).stem}/{subject}").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"./results/{model_name}/{Path(ckpt_path).stem}/{subject}/Step_{j}.png")

    print(f"Mean MSE: {np.mean(mse_list)}")