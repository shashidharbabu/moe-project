from torch import nn
import torch

class StudentRouter(nn.Module):
    def __init__(self):
        super(StudentRouter, self).__init__()
        self.text_encoder = nn.Linear(768, 512)  # CLIP Text Encoder (Frozen)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512, 3)  # Output layer for domain probabilities

    def forward(self, text_input):
        x = self.text_encoder(text_input)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return torch.softmax(x, dim=-1)  # Softmax for domain probabilities

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))  # Load pre-trained weights

    def save_weights(self, path):
        torch.save(self.state_dict(), path)  # Save current weights