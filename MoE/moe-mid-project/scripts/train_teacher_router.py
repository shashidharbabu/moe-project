import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.moe_mid_dataset import MoEMidDataset
from src.models.router_teacher import RouterTeacher
from src.training.router_teacher_trainer import RouterTeacherTrainer
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config('configs/router_teacher.yaml')

    # Set up dataset and dataloader
    dataset = MoEMidDataset(config['data'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model
    model = RouterTeacher(config['model'])
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Initialize trainer
    trainer = RouterTeacherTrainer(model, optimizer, dataloader, config['training'])

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()