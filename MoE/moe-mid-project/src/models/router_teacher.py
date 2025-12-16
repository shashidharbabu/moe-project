from torch import nn
import torch

class TeacherRouter(nn.Module):
    def __init__(self):
        super(TeacherRouter, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.encoder(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)