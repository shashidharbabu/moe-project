from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.data.moe_mid_dataset import MoEMidDataset
from src.models.router_teacher import RouterTeacher
import yaml

class RouterTeacherTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RouterTeacher().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = self._create_data_loader(self.config['train_dataset'])
        
        wandb.init(project="moe-mid-project", config=self.config)

    def _create_data_loader(self, dataset_path):
        dataset = MoEMidDataset(dataset_path)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            wandb.log({"epoch": epoch, "loss": avg_loss})
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    trainer = RouterTeacherTrainer(config_path='configs/router_teacher.yaml')
    trainer.train(epochs=trainer.config['epochs'])
    trainer.save_model(path='models/router_teacher.pt')