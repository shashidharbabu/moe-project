import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.data.moe_mid_dataset import MoE_MID_Dataset
from src.models.expert_unet import ExpertUNet
from src.training.expert_trainer import ExpertTrainer
from src.utils.logging import setup_logging

def main(args):
    # Set up logging
    setup_logging()

    # Load dataset
    dataset = MoE_MID_Dataset(data_dir=args.data_dir, domain=args.domain)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = ExpertUNet()
    model.to(args.device)

    # Initialize trainer
    trainer = ExpertTrainer(model=model, dataloader=dataloader, learning_rate=args.learning_rate)

    # Start training
    trainer.train(num_epochs=args.num_epochs)

    # Save the trained model
    model_save_path = os.path.join(args.output_dir, f"{args.domain}_expert_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Expert Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--domain", type=str, choices=["portrait", "food", "landscape"], required=True, help="Domain for the expert model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")

    args = parser.parse_args()
    main(args)