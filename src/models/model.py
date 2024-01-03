from torch import nn

class myawesomemodel(nn.Module):
    def __init__(self):
        super(myawesomemodel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # [B, 1, 86, 86] -> [B, 32, 84, 84]
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B, 32, 84, 84] -> [B, 32, 42, 42]
            nn.Conv2d(32, 64, 3), # [B, 32, 42, 42] -> [B, 64, 40, 40]
            nn.ReLU(),
            nn.MaxPool2d(2),      # [B, 64, 40, 40] -> [B, 64, 20, 20]
            nn.Flatten(),         # [B, 64, 20, 20] -> [B, 64 * 20 * 20]
            nn.Linear(64 * 20 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),     # Assuming 4 is the number of classes
        )

    def forward(self, x):
        return self.network(x)
