from pathlib import Path
from torch.utils.data import IterableDataset
from sklearn.preprocessing import MinMaxScaler

import torch
import numpy as np
import wfdb

ECG_LENGTH = 650000

# noisy_ecg torch dataset
class NOISY_ECG(IterableDataset):
    def __init__(self, root="./data/physionet.org/files/", channel: list = [0], window_size: int = 1024, train:bool = True):
        super(NOISY_ECG, self).__init__()
        self.root = root
        self.channel = channel
        self.window_size = window_size
        self.train = train

        # MITDB subjects (w/o 118 and 119 who are in NSTDB)
        mitdb_subjects = [s for s in wfdb.io.get_record_list("mitdb") if s not in ["118", "119"]]
        self.val_subjects = np.random.choice(mitdb_subjects, 10)
        self.train_subjects = [s for s in mitdb_subjects if s not in self.val_subjects]

        # pure noise from NSTDB
        self.bw, _ = wfdb.rdsamp(f"{root}/nstdb/1.0.0/bw", channels=channel) # baseline wander
        self.em, _ = wfdb.rdsamp(f"{root}/nstdb/1.0.0/em", channels=channel) # electrode motion artifact
        self.ma, _ = wfdb.rdsamp(f"{root}/nstdb/1.0.0/ma", channels=channel) # muscle artifact
        self.noise_length = min(self.bw.shape[0], self.em.shape[0], self.ma.shape[0])

    def __iter__(self):
        while True:
            random_subject = np.random.choice(self.train_subjects) if self.train else np.random.choice(self.val_subjects)
            signal, _ = wfdb.rdsamp(f"{self.root}/mitdb/1.0.0/{random_subject}", channels=self.channel)

            random_start = np.random.randint(0, ECG_LENGTH % self.window_size)
            clean_signal = signal[random_start:random_start+self.window_size][:, 0]
            noisy_signal = self.add_noise(clean_signal, np.random.choice(["bw", "em", "ma"]), np.random.randint(-6, 24), self.train)
        
            # normalize signal
            scaler = MinMaxScaler()
            noisy_signal = scaler.fit_transform(noisy_signal.reshape(-1, 1)).reshape(-1)
            clean_signal = scaler.fit_transform(clean_signal.reshape(-1, 1)).reshape(-1)

            clean_signal = torch.from_numpy(clean_signal.reshape(1, self.window_size)).float()
            noisy_signal = torch.from_numpy(noisy_signal.reshape(1, self.window_size)).float()

            yield noisy_signal, clean_signal

    def add_noise(self, signal, noise_type, snr, train=True):
        assert noise_type in ["bw", "em", "ma"], "Noise type must be one of ['bw', 'em', 'ma']"

        st = np.random.randint(0, self.noise_length-signal.shape[0]) if train else 0
        if noise_type == "bw":
            noise = self.bw[st:st+signal.shape[0], 0]
        elif noise_type == "em":
            noise = self.em[st:st+signal.shape[0], 0]
        elif noise_type == "ma":
            noise = self.ma[st:st+signal.shape[0], 0]
        
        # mix signal and noise with SNR
        rms_clean = np.sqrt(np.mean(signal**2))
        rms_noise = np.sqrt(np.mean(noise**2))
        scale = rms_clean / (10**(snr / 20) * rms_noise)
        noisy = signal + noise * scale

        return noisy

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    noisy_ecg = NOISY_ECG()
    dataloader = DataLoader(noisy_ecg, batch_size=4, num_workers=0)
    for noisy, clean in dataloader:
        print(noisy.shape, clean.shape)
        break