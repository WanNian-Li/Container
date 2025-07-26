import torch
import torch.nn as nn
import torch.nn.functional as F

class Genal_CNN(nn.Module):
    def __init__(self):
        super(Genal_CNN, self).__init__()

        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        # LazyLinear 自动推断输入维度（第一次 forward 后确定）
        self.fc0 = nn.LazyLinear(768)
        self.bn5 = nn.BatchNorm1d(768)

        self.fc1 = nn.LazyLinear(512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 64)
        self.bn8 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x, global_feature):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.flatten(x)  

        x = self.relu(self.bn5(self.fc0(x)))  # LazyLinear 自动适配
        x = torch.cat([x, global_feature], dim=1)

        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.relu(self.bn8(self.fc3(x)))
        x = self.fc4(x)
        return x

class Genal_LSTM(nn.Module):
    def __init__(self, global_dim=40, hidden_size=64, num_layers=4):
        super(Genal_LSTM, self).__init__()
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        # LSTM input_size will be inferred at runtime
        self.lstm = nn.LSTM(
            input_size=0,  # placeholder
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        # LazyLinear to infer input dim: lstm_out + global_feature
        lstm_out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, global_feature):
        """
        x: shape [batch, 1, stack, tier]
        global_feature: shape [batch, global_dim]
        """
        
        batch_size = x.size(0)    
        seq_len = x.size(2)     # stack
        feature_dim = x.size(3) # tier
        x = x.view(batch_size, seq_len, feature_dim)  # 注意保持 batch_size 不变

        # Rebuild LSTM if input_size is not initialized
        if self.lstm.input_size == 0:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional
            ).to(x.device)

        lstm_out, _ = self.lstm(x)  # shape: [batch, seq_len, hidden*2]
        z = lstm_out[:, -1, :]      # 取最后一个时间步的输出: shape [batch, hidden*2]

        # 拼接全局特征
        z = torch.cat([z, global_feature], dim=1)  # shape: [batch, hidden*2 + global_dim]

        # 全连接
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.fc4(z)  # 最后一层不需要ReLU，输出连续值或回归
        return z
