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

## Project structure


The directory structure of the project looks like this:

```txt

├── .dvc                   <- dvc configuration files
├── .github/workflows      <- CI/CD workflow configurations
├── app                    <- Main application code
│   ├── __init__.py
│   ├── frontend.py        <- Implementation of frontend
│   ├── main.py            <- Main application for predicting tumor type
│   └── main_old.py        <- Older version of the main application
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
│   ├── outputs            <- FJERNES?
│   ├── visualizations     <- FJERNES?
│   ├── data_drifting.py   <- Script to check for data drifting
│   ├── predict_model.py   <- Predicts a tumor type based on a provided raw image
│   ├── sweep.yaml         <- Configuraion file for hyperparameter sweep
│   ├── train_model.py     <- Script for training models
│   └── train_model_lightning.py <- Script for training models with PyTorch Lightning
│
├── tests                  <- Test files
├── .dvcignore             <- DVC ignore file
├── .gitignore             <- Git ignore file
├── .pre-commit-config.yaml<- Pre-commit configuration
├── LICENSE                <- Open-source license if one is chosen
├── Makefile               <- Makefile with convenience commands like `make data` or `make train`
├── README.md              <- The top-level README for developers using this project.
├── requirements.txt       <- The requirements file for reproducing the analysis environment
├── requirements_dev.txt   <- Additional requirements for development
├── requirements_frontend.txt <- Frontend specific requirements
└── requirements_tests.txt <- Testing specific requirements

A LITTLE CLEANUP ON THE SOURCE?


```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
