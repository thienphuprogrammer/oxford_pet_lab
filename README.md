# Oxford Pet Lab - Computer Vision Project

A comprehensive computer vision project for pet detection and segmentation using the Oxford-IIIT Pet Dataset.

## Project Overview

This project implements four main tasks:
1. Dataset exploration and analysis
2. Object detection with/without pre-trained models
3. Semantic segmentation with/without pre-trained models
4. Multitask learning combining detection and segmentation

## Setup Instructions

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate oxford_pet_lab
```

## Project Structure

- `config/`: Configuration files for models and training
- `data/`: Data loading, preprocessing, and augmentation
- `models/`: Model architectures and utilities
- `training/`: Training orchestration and utilities
- `evaluation/`: Model evaluation and metrics
- `utils/`: General utility functions
- `notebooks/`: Jupyter notebooks for each task
- `experiments/`: Experiment tracking and results
- `results/`: Model outputs, logs, and reports
- `tests/`: Unit tests
- `scripts/`: Execution scripts for each task

## Usage

### Task 1: Dataset Exploration
```bash
python scripts/run_task1.py
```

### Task 2: Object Detection
```bash
python scripts/run_task2.py
```

### Task 3: Semantic Segmentation
```bash
python scripts/run_task3.py
```

### Task 4: Multitask Learning
```bash
python scripts/run_task4.py
```

### Generate Final Report
```bash
python scripts/generate_report.py
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- matplotlib
- numpy
- pandas
- scikit-learn
- jupyter

## License

MIT License

