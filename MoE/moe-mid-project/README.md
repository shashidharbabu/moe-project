# MoE-MID Project

## Overview
The Mixture-of-Experts for Multi-Image Diffusion (MoE-MID) project aims to develop a unified diffusion model that effectively generalizes across diverse visual domains, including Portraits, Food, and Landscapes. By leveraging a Mixture-of-Experts (MoE) architecture, the project seeks to preserve fine-grained style and prevent attribute leakage during image generation.

## Project Structure
The project is organized into several key directories and files:

- **configs/**: Contains configuration files for baseline and expert model training, as well as router and inference settings.
- **data/**: Holds raw datasets and processed metadata.
- **docs/**: Documentation detailing the architecture, training pipeline, and evaluation plan.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and router performance analysis.
- **scripts/**: Python scripts for downloading datasets, preprocessing images, training models, and evaluating performance.
- **src/**: Source code for the project, including model definitions, training logic, evaluation metrics, and the inference pipeline.
- **tests/**: Unit tests for various components of the project.
- **tools/**: Tools for prompt sets and evaluation suites.

## Setup Instructions
To set up the project environment, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd moe-mid-project
   ```

2. **Create a Conda environment**:
   ```
   conda env create -f environment.yml
   conda activate moe-mid
   ```

3. **Install additional requirements**:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
### Inference Pipeline
To use the inference pipeline, follow these steps:

1. **Start the FastAPI application**:
   ```
   uvicorn src.api.fastapi_app:app --reload
   ```

2. **Send a POST request to the inference endpoint** with a JSON object containing your prompt:
   ```json
   {
       "prompt": "A beautiful landscape"
   }
   ```

3. **Receive the generated image** in response.

### Training Models
To train the models, use the provided scripts in the `scripts/` directory. For example, to train the baseline model:
```
python scripts/train_baseline.py --config configs/baseline.yaml
```

## Documentation
For detailed information on the architecture, training procedures, and evaluation metrics, refer to the documentation files located in the `docs/` directory.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.