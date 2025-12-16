import torch

def compute_delta_weights(expert_model_path, base_model_path, delta_weight_path):
    # Load the expert model weights
    expert_model = torch.load(expert_model_path)
    
    # Load the base model weights
    base_model = torch.load(base_model_path)
    
    # Calculate delta weights
    delta_weights = {}
    for key in expert_model.keys():
        delta_weights[key] = expert_model[key] - base_model[key]
    
    # Save the delta weights
    torch.save(delta_weights, delta_weight_path)

if __name__ == "__main__":
    # Define paths for the models
    expert_model_path = "path/to/expert/model.pt"  # Update with actual path
    base_model_path = "path/to/base/model.pt"      # Update with actual path
    delta_weight_path = "path/to/save/delta_weights.pt"  # Update with actual path

    # Compute and save delta weights
    compute_delta_weights(expert_model_path, base_model_path, delta_weight_path)