import os
import requests
from tqdm import tqdm

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(
        desc=dest,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)

def download_datasets():
    datasets = {
        "FFHQ": "https://example.com/path/to/ffhq.zip",
        "Food-101": "https://example.com/path/to/food101.zip",
        "Places365": "https://example.com/path/to/places365.zip"
    }

    for name, url in datasets.items():
        print(f"Downloading {name}...")
        dest = os.path.join('data', 'raw', name.lower().replace('-', '').replace(' ', ''), f"{name}.zip")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        download_file(url, dest)
        print(f"{name} downloaded to {dest}")

if __name__ == "__main__":
    download_datasets()