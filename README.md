# MLOps-Project 
## Image Brain Tumor Classification (MRI)

Exam project for course 02476 Machine Learning Operations at the Technical University of Denmark (DTU)

### Project Goal
The goal of this project is to develop an efficient machine learning model capable of predicting types of brain tumors from MRI images. The emphasis is on delivering a functional solution, focusing on MLOps practices with the entire pipeline, from data acquisition, model training, and deployment to monitoring and optimization. 

### Framework
For the prediction of brain tumors, we have chosen the TIMM framework, specifically tailored for Computer Vision tasks in PyTorch.
We will use this framework to construct a deep learning model and include the framework as part of our environment.

### Data
This model is trained on the [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. It consists of 3264 images in jpg format (93.08 MB). The dataset is divided into a training (88%) and a test dataset (12%), each divided into four types of brain tumors: glioma, meningioma, pituitary, or no tumor.

### Models
The project aims to explore a deep learning architectures for image classification, focusing on convolutional neural networks (CNNs). We use the ResNet-18 architecture, available through the TIMM (PyTorch Image Models) framework, for image classification. The choice is driven by TIMM's extensive collection of pre-trained models, with ResNet-18 offering a balanced trade-off between model complexity and computational efficiency.

## Project structure


The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── MLOps-Project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
