from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
import yaml
import os
from src.data.moe_mid_dataset import MoEMidDataset
from src.models.expert_unet import ExpertUNet

class ExpertTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.num_epochs = self.config['num_epochs']
        self.dataset_path = self.config['dataset_path']
        self.model_save_path = self.config['model_save_path']

        self.model = ExpertUNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        wandb.init(project="moe-mid", config=self.config)

    def load_data(self):
        dataset = MoEMidDataset(self.dataset_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        dataloader = self.load_data()
        for epoch in range(self.num_epochs):
            for images, _ in dataloader:
                images = images.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)  # Assuming a reconstruction task
                loss.backward()
                self.optimizer.step()

                wandb.log({"loss": loss.item()})

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")

        self.save_model()

    def save_model(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        model_path = os.path.join(self.model_save_path, 'expert_unet.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    trainer = ExpertTrainer(config_path='configs/experts.yaml')
    trainer.train()