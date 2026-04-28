# PawPularity Contest

A picture is worth a thousand words. But did you know a picture can save a thousand lives? Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. You might expect pets with attractive photos to generate more interest and be adopted faster. But what makes a good picture? With the help of data science, you may be able to accurately determine a pet photo’s appeal and even suggest improvements to give these rescue animals a higher chance of loving homes.

PetFinder.my is Malaysia’s leading animal welfare platform, featuring over 180,000 animals with 54,000 happily adopted. PetFinder collaborates closely with animal lovers, media, corporations, and global organizations to improve animal welfare.

Currently, PetFinder.my uses a basic Cuteness Meter to rank pet photos. It analyzes picture composition and other factors compared to the performance of thousands of pet profiles. While this basic tool is helpful, it's still in an experimental stage and the algorithm could be improved.

In this competition, you’ll analyze raw images and metadata to predict the “Pawpularity” of pet photos. You'll train and test your model on PetFinder.my's thousands of pet profiles. Winning versions will offer accurate recommendations that will improve animal welfare.

If successful, your solution will be adapted into AI tools that will guide shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements. As a result, stray dogs and cats can find their "furever" homes much faster. With a little assistance from the Kaggle community, many precious lives could be saved and more happy families created.

Top participants may be invited to collaborate on implementing their solutions and creatively improve global animal welfare with their AI skills.

# Summary 

Uses the Kaggle My Petfinder dataset Builds a pipeline to: load and analyze image data using a CNN model.

# Dataset

Dataset: ~9000 image data samples

Competition: PetFinder.my - Pawpularity Contest (Kaggle)

# Approach

Pipeline stages: Data preprocessing Model training Evaluation Kaggle submission generation

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
