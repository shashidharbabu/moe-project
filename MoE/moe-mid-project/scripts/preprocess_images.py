import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, size=(512, 512)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_metadata = []

    for domain in os.listdir(input_dir):
        domain_path = os.path.join(input_dir, domain)
        if os.path.isdir(domain_path):
            for img_name in tqdm(os.listdir(domain_path), desc=f'Processing {domain}'):
                img_path = os.path.join(domain_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    output_path = os.path.join(output_dir, f"{domain}_{img_name}")
                    cv2.imwrite(output_path, img)

                    image_metadata.append({
                        'filename': f"{domain}_{img_name}",
                        'domain': domain
                    })

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(image_metadata, f, indent=4)

if __name__ == "__main__":
    input_directory = '../data/raw'  # Adjust as necessary
    output_directory = '../data/processed'  # Adjust as necessary
    preprocess_images(input_directory, output_directory)