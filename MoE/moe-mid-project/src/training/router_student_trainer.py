from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.moe_mid_dataset import MoEMidDataset
from tqdm import tqdm
import wandb

class StudentRouterTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.train_loader = self._initialize_data_loader()

    def _initialize_model(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        model.to(self.device)
        return model

    def _initialize_data_loader(self):
        dataset = MoEMidDataset(self.config['dataset_path'])
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train(self, epochs):
        wandb.init(project="moe-mid-project", config=self.config)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                captions, soft_labels = batch
                captions = captions.to(self.device)
                soft_labels = soft_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(captions).logits_per_image
                loss = self.loss_fn(outputs, soft_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            wandb.log({"loss": avg_loss})

        wandb.finish()

if __name__ == "__main__":
    import yaml

    with open("configs/router_student.yaml", 'r') as file:
        config = yaml.safe_load(file)

    trainer = StudentRouterTrainer(config)
    trainer.train(epochs=config['epochs'])