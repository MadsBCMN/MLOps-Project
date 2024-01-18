# MLOps-Project 
## Image Classification of Brain Tumors from MRI

Exam project for course 02476 Machine Learning Operations at the Technical University of Denmark (DTU).

### Project Goal
The goal of this project is to develop an efficient machine learning model capable of predicting types of brain tumors from MRI images. The emphasis is on delivering a functional solution, focusing on MLOps practices with the entire pipeline, from data acquisition, model training, and deployment to monitoring and optimization. 

### Framework
For the prediction of brain tumors, we have chosen the TIMM framework, specifically tailored for Computer Vision tasks in PyTorch.
We will use this framework to construct a deep learning model and include the framework as part of our environment.

### Data
This model is trained on the [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. It consists of 3264 images in jpg format (93.08 MB). The dataset is divided into a training (88%) and a test dataset (12%), each divided into four types of brain tumors: glioma, meningioma, pituitary, or no tumor.

### Models
The project aims to explore a deep learning architectures for image classification, focusing on convolutional neural networks (CNNs). We use the ResNet-18 architecture, available through the TIMM (PyTorch Image Models) framework, for image classification. The choice is driven by TIMM's extensive collection of pre-trained models, with ResNet-18 offering a balanced trade-off between model complexity and computational efficiency.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- Docker (for containerization)
- Access to command line interface

## Using the application

In order to use the MRI-classification model, follow these steps to run the model locally.

Clone the repository to your local machine:

```bash
git clone https://github.com/MadsBCMN/MLOps-Project.git
```

Go to the root directory:

```bash
cd MLOps-Project
```

Create a conda environment:

```bash
conda create -n myenv python=3.11 && conda activate myenv
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Get the data:

```bash
dvc pull
```

Run the following command to start the predict API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Change backend = http://localhost:8000/classify" in frontend.py and run the following command:

```bash
streamlit run app/frontend.py
```

This will open the application in our browser. You are now able to upload an image of a MRI brain-scan. When you click "Classify MRI Scan", the application will return the predicted tumor category for the image.


The application can be accessed directly in the web-browser using the following link:

https://frontend-3aoiym5c7a-lz.a.run.app/


## Visualizations

Follow this link to access training report from wandb

https://api.wandb.ai/links/mlops_project_team/0szdkbjq

## Project structure


The directory structure of the project looks like this:

```txt

├── .dvc                   <- dvc configuration files
├── .github/workflows      <- CI/CD workflow configurations
├── app                    <- Main application code
│   ├── __init__.py
│   ├── frontend.py        <- Implementation of frontend
│   └──  main.py            <- Main application for predicting tumor type
│
├── data                   <- Data used for analysis and modeling
│   ├── processed.dvc      <- Pointer to remote storage of processed images for training and testing
│   ├── raw.dvc            <- Pointer to raw images of tumors
│   ├── data_features      <- Images to check for data drifting
│   └── example_images     <- Images for testing the application
│
├── dockerfiles            <- Dockerfiles for containerization
├── outputs                <- Output files from Hydra
├── reports                <- Folder containing all report related files
│   ├── README.md          <- Markdown file containing the report
│   ├── report.py          <- Test script for the report
│   └── figures            <- Figures for the report
│
├── src                    <- Source code for use in this project.
│   ├── __init__.py        
│   ├── config             <- Folder with configuration files for models, training, etc.
│   ├── data               <- Scripts to process and load data
│   │   ├── __init__.py
│   │   ├── config.py      <- Configure standard image size
│   │   ├── dataloader.py  <- Script for loading the data into a Pytorch Dataset
│   │   ├── make_dataset.py<- Processes and converts raw images into tensors
│   │   └── unpack_data.py <- File for unpacking data stored as ZIP
│   │
│   ├── models             <- Scripts and modules related to the ML models
│   │   ├── __init__.py
│   │   └── model.py       <- Function for returning the model architecture
│   │
│   ├── outputs            <- Stored Hydra outputs
│   ├── data_drifting.py   <- Script to check for data drifting
│   ├── predict_model.py   <- Predicts a tumor type based on a provided raw image
│   ├── sweep.yaml         <- Configuraion file for hyperparameter sweep
│   ├── train_model.py     <- Script for training models
│   └── train_model_lightning.py <- Script for training models with PyTorch Lightning
│
├── tests                  <- Test files
├── .dvcignore             <- Specifies files that DVC should ignore.
├── .gitignore             <- Specifies intentionally untracked files to ignore in Git.
├── .pre-commit-config.yaml<- Configuration file for pre-commit hooks to standardize code.
├── LICENSE                <- The license for the project.
├── Makefile               <- Makefile with commands to facilitate project tasks.
├── README.md              <- The top-level README for developers using this project.
├── __init__.py            <- Initialization script that can turn folders into Python packages.
├── cloudbuild-predict.yaml<- Google Cloud Build configuration for prediction tasks.
├── cloudbuild-run-frontend.yaml <- Cloud Build configuration for running the frontend.
├── cloudbuild-run-predict.yaml <- Cloud Build configuration for running predictions.
├── cloudbuild-run-training.yaml <- Cloud Build configuration for training tasks.
├── cloudbuild-training.yaml <- Cloud Build configuration for setting up training environments.
├── cloudbuild.yaml        <- Main Cloud Build configuration file.
├── clouddeploy-run-predict.yaml <- Cloud Deploy configuration for prediction services.
├── models.dvc             <- DVC file to manage versions of models.
├── profiler.dvc           <- DVC file related to profiling tools or output.
├── pyproject.toml         <- Configuration file for Python project settings.
├── requirements.txt       <- The requirements file for reproducing the analysis environment.
├── requirements_dev.txt   <- The requirements file for development environments.
├── requirements_frontend.txt <- Requirements file for frontend-specific dependencies.
└── requirements_tests.txt <- Requirements file for testing environments.


```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
