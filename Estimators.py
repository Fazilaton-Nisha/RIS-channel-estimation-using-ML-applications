import torch
import torch.nn as nn
import torch.nn.functional as F


# the DCE network for Pilot_num 128
class DCE_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.kernel_size = 3
        self.padding = 1
        layers = []
        # the first layer
        layers.append(nn.Conv2d(in_channels=2, out_channels=self.features, kernel_size=self.kernel_size, stride=1,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.BatchNorm2d(self.features))
        layers.append(nn.ReLU(inplace=True))

        # the second and the third layer
        for i in range(2):
            layers.append(
                nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size, stride=1,
                          padding=self.padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(self.features))
            layers.append(nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(*layers)

        # the linear layer
        self.FC = nn.Linear(self.features * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(x.shape[0], self.features * 16 * 8)
        # print(x.shape)
        x = self.FC(x)
        return x

# the scenario classifier for Pilot_num 128
class SC_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                               bias=False)
        self.FC = nn.Linear(32 * 4 * 2, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(x.shape[0], 32 * 4 * 2)
        # print(x.shape)
        x = self.FC(x)
        return F.log_softmax(x, dim=1)

# the feature extractor for Pilot_num 128
class Conv_P128(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.kernel_size = 3
        self.padding = 1
        layers = []
        # the first layer
        layers.append(
            nn.Conv2d(in_channels=2, out_channels=self.features, kernel_size=self.kernel_size, padding=self.padding,
                      bias=False))
        layers.append(nn.BatchNorm2d(self.features))
        layers.append(nn.ReLU(inplace=True))

        # the second and the third layer
        for i in range(2):
            # the second layer
            layers.append(nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.features))
            layers.append(nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.shape)
        x = x.view(x.shape[0], self.features * 16 * 8)

        return x

# the feature mapper for Pilot_num 128
class FC_P128(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(32 * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.FC(x)
        return x

def NMSE_cuda(x_hat, x):
    power = torch.sum(x**2)
    mse = torch.sum((x_hat - x)**2)
    nmse = mse/power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)

        return nmse