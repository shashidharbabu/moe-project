from scipy.linalg import sqrtm
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from sklearn.metrics import pairwise

def calculate_fid(real_images, generated_images):
    # Load Inception model
    model = inception_v3(pretrained=True, transform_input=False).eval()
    
    # Preprocess images
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    real_images = torch.stack([preprocess(img) for img in real_images])
    generated_images = torch.stack([preprocess(img) for img in generated_images])
    
    with torch.no_grad():
        real_features = model(real_images).detach().numpy()
        generated_features = model(generated_images).detach().numpy()
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(cov_real + cov_gen - 2 * sqrtm(np.dot(cov_real, cov_gen)))
    return fid

def calculate_kid(real_images, generated_images, n_subsets=100):
    # Calculate the kernel Inception distance (KID)
    real_features = []
    generated_features = []
    
    for _ in range(n_subsets):
        real_subset = np.random.choice(real_images, size=len(real_images) // n_subsets, replace=False)
        generated_subset = np.random.choice(generated_images, size=len(generated_images) // n_subsets, replace=False)
        
        real_features.append(calculate_fid(real_subset, generated_subset))
        generated_features.append(calculate_fid(generated_subset, real_subset))
    
    # KID is the average of the squared differences
    kid = np.mean(np.square(np.array(real_features) - np.array(generated_features)))
    return kid