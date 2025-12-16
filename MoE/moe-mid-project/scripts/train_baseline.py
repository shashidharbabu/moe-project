import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.moe_mid_dataset import MoEMidDataset
from src.training.baseline_trainer import BaselineTrainer

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    dataset = MoEMidDataset(config['data'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize trainer
    trainer = BaselineTrainer(config['model'], device)

    # Start training
    trainer.train(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to the config file.')
    args = parser.parse_args()

    main(args.config)