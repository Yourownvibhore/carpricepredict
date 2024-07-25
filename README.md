# Used Car Price Prediction

## Overview

This repository contains the code and resources for an end-to-end Used Car Price Prediction project. The goal of this project is to predict the price of used cars based on various features such as model, brand, year, mileage, and more.
the project is live at https://usedcarprice.onrender.com/

## Project Structure

- **data/**
  - `raw_data.csv`          # Raw dataset containing used car information
  - `processed_data.csv`    # Processed dataset used for model training

- **notebooks/**
  - `used_car.ipynb`  # Jupyter notebook for exploring the dataset and using different model

- **src/**
  - **components/**
     - `data_ingestion.py`
     - `data_transformation.py`
     - `model_trainer.py`
   - **pipeline/**
     - `predict_pipeline.py`
     - `train_pipeline.py`
  - `logger.py`  # logger functions used throughout the project
  - `exception.py`  # exception handling functions used throughout the project
  - `utils.py`  # Utility functions used throughout the project
  - `setup.py`  # information about the project

- **app/**
  - `app.py`  # Flask web application for predicting used car prices
  - **templates/**
    - `about.html`
    - `blog.html`
    - `contact.html`
    - `home.html`
    - `prediction.html`


- `requirements.txt`  # List of dependencies for the project
- `config.yaml`  # Configuration file for setting up model parameters
- `README.md`  # Project documentation

Contributing
If you would like to contribute to this project, please follow the standard GitHub workflow:
# CarPrice
