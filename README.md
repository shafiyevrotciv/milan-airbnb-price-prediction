# Airbnb Milan Price Prediction

## Project Overview
This project aims to predict Airbnb rental prices in Milan using machine learning. It demonstrates a complete end-to-end data science workflow, including exploratory data analysis (EDA), feature engineering, hyperparameter tuning, and the deployment of a robust model ensemble (XGBoost, LightGBM, and CatBoost).

## Dataset
The dataset used in this project is sourced from [Inside Airbnb](http://insideairbnb.com/).
To run the pipeline locally:
1. Download the Milan listings dataset (`listings.csv`).
2. Place it into the `data/` directory. 

## Project Structure
```text
├── data                  # Directory for raw data and predictions (ignored by Git)
├── models                # Directory for saved models and preprocessor (ignored by Git)
├── notebooks
│   └── Milan_AirBnb.ipynb # EDA, feature selection, and Optuna analysis
├── src
│   ├── preprocess.py      # Data cleaning and feature engineering pipeline
│   ├── train.py           # Model training and ensemble weighting script
│   └── predict.py         # Command-line inference script
├── .gitignore
├── requirements.txt
└── README.md
```


## Installation
To run this project locally, clone the repository and install dependencies:

```bash
git clone https://github.com/shafiyevrotciv/milan-airbnb-price-prediction.git
cd milan-airbnb-price-prediction
pip install -r requirements.txt
```
## Usage

### 1. Train the Model Pipeline
Runs data preprocessing, fits transformers, optimizes hyperparameters via Optuna, and saves the trained models to the models/ directory:
```bash
python src/train.py
```
### 2. Run Inference
Generate predictions for any new CSV dataset using the pre-trained ensemble:
```bash
python src/predict.py --input data/listings.csv --output data/predictions.csv
```