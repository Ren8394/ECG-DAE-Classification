import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.blstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, in_dim, bias=True)

    def forward(self, x):
        y, _ =self.blstm(x)
        y = self.fc(y)
        return y
    
if __name__ == "__main__":
    x = torch.randn(4, 1, 1024)
    model = BLSTM()
    y = model(x)
    print(y.shape)