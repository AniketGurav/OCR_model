import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, eow, recursions=1):
        super().__init__()
        
        self.eow = eow
        self.recursions = recursions
        
        self.conv1_0 = nn.Conv2d(1, 64, (3, 3), padding="same")
        self.conv1_t = nn.Conv2d(64, 64, (3, 3), padding="same")

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)  # -> [1, 64, 16, 50]

        self.conv2_0 = nn.Conv2d(64, 128, (3, 3), padding="same")
        self.conv2_t = nn.Conv2d(128, 128, (3, 3), padding="same")

        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv3_0 = nn.Conv2d(128, 256, (3, 3), padding="same")
        self.conv3_t = nn.Conv2d(256, 256, (3, 3), padding="same")

        self.pool3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv4_0 = nn.Conv2d(256, 512, (3, 3), padding="same")
        self.conv4_t = nn.Conv2d(512, 512, (3, 3), padding="same")

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(24576, 1024)
        self.dense2 = nn.Linear(1024, 1024) # -> [1, 1024]

        self.rnn_1 = nn.LSTM(input_size=1024, hidden_size=256,
                             num_layers=1, batch_first=True,
                             proj_size=self.eow.size(0)) #lstm

        # self.attention = 

        self.rnn_2 = nn.LSTM(input_size=101, hidden_size=256,
                             num_layers=1, batch_first=True,
                             proj_size=self.eow.size(0))
    
    def forward(self, x):
        x = self.conv1_0(x)
        for _ in range(self.recursions):
            x = self.conv1_t(x)
        
        x = self.pool1(x)
        
        
        x = self.conv2_0(x)
        for _ in range(self.recursions):
            x = self.conv2_t(x)
        
        x = self.pool2(x)
        
        x = self.conv3_0(x)
        for _ in range(self.recursions):
            x = self.conv3_t(x)
        
        x = self.pool3(x)
        
        x = self.conv4_0(x)
        for _ in range(self.recursions):
            x = self.conv4_t(x)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.dense2(x)
        
        x = torch.unsqueeze(x, dim=1)

        h0 = torch.autograd.Variable(torch.zeros(1, 1, self.eow.size(0)))
        c0 = torch.autograd.Variable(torch.zeros(1, 1, 256))
        x, (hn, cn) = self.rnn_1(x, (h0, c0))
        
        x, (hn, cn) = self.rnn_2(x, (h0, c0))
        
        return x