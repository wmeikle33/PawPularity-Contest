# Summary 

This project predicts PetFinder Pawpularity scores from pet images using a convolutional neural network. The pipeline includes preprocessing, model training, evaluation, and prediction generation for the Kaggle competition.

# Dataset

Dataset: ~9000 image data samples

Competition: PetFinder.my - Pawpularity Contest (Kaggle)

# Approach

Pipeline stages: Data preprocessing Model training Evaluation Kaggle submission generation.

# Preprocessing

Image is decoded and the data is compiled in a csv.

# Modeling

This repo utilizes a CNN based model to make predictions for the Pawpularity score of a given photo.

# Quickstart

```
git clone https://github.com/wmeikle33/PawPularity-Contest.git
cd PawPularity-Contest
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mkdir -p data/raw

python scripts/train.py
python scripts/predict.py
pytest

```

# Repo Structure

```bash

Pawpularity/
├── pyproject.toml
├── pre_commit_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── src/
│   └── pawpularity_project/
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── data.py
├── scripts/
│   ├── train.py
│   └── predict.py
└── tests/


```


# Modeling

The repository incorporates a CNN model for image analysis. Specifically the model uses a number of max pooling and dense layers to extract image informaiton for analysis.

# Reproduce My Score

```bash

# Reproduce the baseline

1. Clone the repo
2. Create a virtual environment
3. Install dependencies
4. Put `train.csv` and `test.csv` in `data/raw/`
5. Train the baseline model
6. Generate predictions

```bash
git clone https://github.com/wmeikle33/Pawpularity-Contest.git
cd Pawpularity-Contest

python -m venv .venv
source .venv/bin/activate
pip install -e ".[data]"

mkdir -p data/raw
This repository was originally generated from the notebook Pawpularity Contest

```
