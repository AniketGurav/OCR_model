import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, eow, recursions=1):
        super().__init__()
        
        self.sow_size = eow.size(0)  # SOW = 0, 0, ..., 0, 0
        self.eow = eow  # EOW = 0, 0, ..., 0, 1
        self.recursions = recursions
        
        self.conv1_0 = nn.Conv2d(1, 64, (3, 3), padding="same")
        self.conv1_t = nn.Conv2d(64, 64, (3, 3), padding="same")

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2_0 = nn.Conv2d(64, 128, (3, 3), padding="same")
        self.conv2_t = nn.Conv2d(128, 128, (3, 3), padding="same")

        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv3_0 = nn.Conv2d(128, 256, (3, 3), padding="same")
        self.conv3_t = nn.Conv2d(256, 256, (3, 3), padding="same")

        self.pool3 = nn.MaxPool2d((2, 2), stride=2)

        self.conv4_0 = nn.Conv2d(256, 512, (3, 3), padding="same")
        self.conv4_t = nn.Conv2d(512, 512, (3, 3), padding="same")

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(24576, 4096)
        self.dense2 = nn.Linear(4096, 4096)

        self.rnn_1 = nn.LSTM(input_size=self.sow_size, hidden_size=1024,
                             num_layers=1, batch_first=True,
                             proj_size=self.eow.size(0))

        self.attention = EnergyAttention(4096, self.eow.size(0))

        self.rnn_2 = nn.LSTM(input_size=4096, hidden_size=1024,
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

        I = torch.unsqueeze(x, dim=1)

        batch_size = x.size(0)
        batched_sow = torch.autograd.Variable(torch.zeros(size=(x.size(0), 1, self.sow_size))).to(device)
        h0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.eow.size(0))).to(device)
        c0 = torch.autograd.Variable(torch.zeros(1, batch_size, 1024)).to(device)
        results = torch.autograd.Variable(torch.zeros(batch_size, self.sow_size, 23)).to(device)

        s, (hn_1, cn_1) = self.rnn_1(batched_sow, (h0, c0))
        c_t = self.attention(I, s)
        x, (hn_2, cn_2) = self.rnn_2(c_t, (h0, c0))
        results[:, :, 0] = torch.squeeze(x, dim=1)
        for idx in range(1, 23):
            s, (hn_1, cn_1) = self.rnn_1(x, (hn_1, cn_1))
            c_t = self.attention(I, s)
            x, (hn_2, cn_2) = self.rnn_2(c_t, (hn_2, cn_2))
            results[:, :, idx] = torch.squeeze(x, dim=1)

        return results


class EnergyAttention(nn.Module):
    def __init__(self, I_shape, s_shape):
        super().__init__()

        self.s_shape = s_shape
        self.out_shape = I_shape
        
        self.s_layer = nn.Linear(self.s_shape, self.out_shape)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, I, s):
        s_proj = self.s_layer(s)
        result = self.tanh(I + s_proj)  # -> [1, 1, 1024]
        
        alpha = self.softmax(result)  # -> [1, 1, 1024]
        return alpha * I