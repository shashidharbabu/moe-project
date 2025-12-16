# Makefile for MoE-MID Project

.PHONY: all setup data preprocess train_baseline train_expert train_teacher_router train_student_router evaluate clean

all: setup data preprocess train_baseline train_expert train_teacher_router train_student_router evaluate

setup:
	@echo "Setting up the environment..."
	conda env create -f environment.yml

data:
	@echo "Downloading datasets..."
	python scripts/download_datasets.py

preprocess:
	@echo "Preprocessing images..."
	python scripts/preprocess_images.py

train_baseline:
	@echo "Training baseline model..."
	python scripts/train_baseline.py

train_expert:
	@echo "Training expert models..."
	python scripts/train_expert.py

train_teacher_router:
	@echo "Training teacher router model..."
	python scripts/train_teacher_router.py

train_student_router:
	@echo "Training student router model..."
	python scripts/train_student_router.py

evaluate:
	@echo "Evaluating models..."
	python scripts/evaluate_models.py

clean:
	@echo "Cleaning up..."
	rm -rf data/processed/*