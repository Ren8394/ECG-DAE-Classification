import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_DAE(nn.Module):
    def __init__(self, in_dim=1024, latent_dim=32, use_bn=True):
        super(FCN_DAE, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),   
            nn.MaxPool1d(2),                                       
            nn.BatchNorm1d(16) if use_bn else nn.Identity(),                                     
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool1d(2),                                      
            nn.BatchNorm1d(32) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool1d(2),                               
            nn.BatchNorm1d(32) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool1d(2),                             
            nn.BatchNorm1d(16) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool1d(2), 
            nn.BatchNorm1d(1) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.Flatten(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(1) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.ConvTranspose1d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(1) if use_bn else nn.Identity(),
            nn.ELU(),
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
        )
    
        self._init_weights()

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, f"Input dimension mismatch. Expected {self.in_dim}, got {x.shape[-1]}"
        latent = self.encoder(x)
        z = latent.view(-1, 1, self.latent_dim)
        y = self.decoder(z)
        
        return y, latent
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = FCN_DAE(use_bn=False)
    x = torch.randn(1, 1, 1024)
    y, latent = model(x)
    print(y.shape, latent.shape)