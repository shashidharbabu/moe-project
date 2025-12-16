from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.moe_mid_dataset import MoEMidDataset
from src.models.base_components import StableDiffusionModel
from src.utils.logging import setup_logging

class BaselineTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.model = StableDiffusionModel()
        self.dataset = self.setup_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.logger = setup_logging()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return MoEMidDataset(self.config['data_paths'], transform=transform)

    def train(self):
        self.model.train()
        for epoch in range(self.config['num_epochs']):
            for batch in self.dataloader:
                images, labels = batch
                self.model.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.model.loss_function(outputs, labels)
                loss.backward()
                self.model.optimizer.step()
                self.logger.info(f'Epoch [{epoch+1}/{self.config["num_epochs"]}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    trainer = BaselineTrainer(config_path='configs/baseline.yaml')
    trainer.train()