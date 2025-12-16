import torch

def merge_weights(base_weights_path, delta_weights_paths, domain_weights):
    """
    Merges the base U-Net weights with the delta weights based on the provided domain weights.

    Parameters:
    - base_weights_path (str): Path to the base U-Net weights.
    - delta_weights_paths (list of str): List of paths to the delta weights for each domain.
    - domain_weights (list of float): List of weights corresponding to each domain.

    Returns:
    - merged_weights (torch.Tensor): The merged U-Net weights.
    """
    # Load the base U-Net weights
    base_weights = torch.load(base_weights_path)

    # Initialize the merged weights with the base weights
    merged_weights = base_weights.clone()

    # Merge the delta weights based on the domain weights
    for delta_weights_path, weight in zip(delta_weights_paths, domain_weights):
        delta_weights = torch.load(delta_weights_path)
        merged_weights += weight * delta_weights

    return merged_weights

if __name__ == "__main__":
    # Example usage
    base_weights_path = "path/to/base_weights.pt"
    delta_weights_paths = [
        "path/to/portrait_delta.pt",
        "path/to/food_delta.pt",
        "path/to/landscape_delta.pt"
    ]
    domain_weights = [0.5, 0.3, 0.2]  # Example weights for each domain

    merged_weights = merge_weights(base_weights_path, delta_weights_paths, domain_weights)
    torch.save(merged_weights, "path/to/merged_weights.pt")