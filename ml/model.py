import torch 
import torch.nn as nn
import torch.nn.functional as F

class TinyRes28(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # --- Initial Layer (28x28) ---
        # Stride 1 to keep resolution high for the index finger
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 1: 32 -> 64 (28x28)
        self.dw1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.pw1 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.bn1_out = nn.BatchNorm2d(64)

        # Pool to 14x14
        self.pool = nn.MaxPool2d(2)

        # Block 2: 64 -> 128 (14x14)
        self.dw2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False)
        self.pw2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn2_out = nn.BatchNorm2d(128)

        # Block 3: Residual 128 -> 128 (14x14)
        self.dw3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False)
        self.pw3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn3_out = nn.BatchNorm2d(128)

        # Block 4: Residual 128 -> 128 (14x14)
        self.dw4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False)
        self.pw4 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn4_out = nn.BatchNorm2d(128)

        # Final Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 28, 28)
        
        # Initial
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Block 1
        x = F.relu(self.bn1_out(self.pw1(F.relu(self.dw1(x)))))
        x = self.pool(x) # 14x14

        # Block 2
        x = F.relu(self.bn2_out(self.pw2(F.relu(self.dw2(x)))))

        # Block 3 (Skip Connection)
        identity = x
        x = F.relu(self.dw3(x))
        x = self.bn3_out(self.pw3(x))
        x = F.relu(x + identity)

        # Block 4 (Skip Connection)
        identity = x
        x = F.relu(self.dw4(x))
        x = self.bn4_out(self.pw4(x))
        x = F.relu(x + identity)

        # Head
        x = self.gap(x)
        x = torch.flatten(x, 1) # (Batch, 128)
        x = self.dropout(x)
        return self.fc(x)