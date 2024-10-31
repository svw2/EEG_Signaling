import torch
import torch.nn as nn

class NWK(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.c1 = nn.Conv2d(1,8,3,2,1)
        self.c2 = nn.Conv2d(8,16,3,2,1)
        self.c3 = nn.Conv2d(16,32,3,2,1)
        self.L1 = nn.Linear(512, 2048)
        self.L2 = nn.Linear(2048, output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self. L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.tanh(x)*1.5
        return x


class NWK2(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.c1 = nn.Conv1d(input_dim[0],16,3,1,1)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(2,2)
        self.c2 = nn.Conv1d(16,64,3,1,1)

    def forward(self,x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.mp(x)

        return x 





