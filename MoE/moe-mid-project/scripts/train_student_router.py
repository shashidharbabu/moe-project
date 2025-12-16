import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.moe_mid_dataset import MoEMidDataset
from src.models.router_student import StudentRouter
from src.training.utils import set_seed
import wandb
import yaml

def train_student_router(config):
    # Set random seed for reproducibility
    set_seed(config['seed'])

    # Load dataset
    dataset = MoEMidDataset(config['data']['path'], transform=config['data']['transform'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model
    model = StudentRouter()
    model.train()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.KLDivLoss(reduction='batchmean')

    # Initialize Weights and Biases for logging
    wandb.init(project="moe-mid-project", config=config)

    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for batch in dataloader:
            captions, soft_labels = batch

            # Forward pass
            outputs = model(captions)
            loss = criterion(outputs, soft_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{config['training']['epochs']}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), config['model']['save_path'])

if __name__ == "__main__":
    with open("configs/router_student.yaml", 'r') as file:
        config = yaml.safe_load(file)
    train_student_router(config)