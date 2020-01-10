import torch.nn as nn
import torch.nn.functional as F

fc1_out = 32
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, fc1_out)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc1_out, fc1_out)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(fc1_out, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)