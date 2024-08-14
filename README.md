# Heart Disease Prediction

This repository contains a machine learning project for predicting heart disease. The project utilizes several advanced techniques and models to provide accurate predictions based on a dataset of heart-related health metrics.

## Project Overview

The goal of this project is to build a predictive model for heart disease using machine learning techniques. The dataset includes various features such as age, sex, cholesterol levels, and other relevant health indicators.

## Key Features

- **Data Preprocessing**: Includes handling missing values, encoding categorical variables, and scaling features.
- **Dimensionality Reduction**: Uses Principal Component Analysis (PCA) to reduce the number of features while retaining most of the variance in the dataset.
- **Model Selection and Tuning**: Implements and tunes several machine learning models, including:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- **Hyperparameter Optimization**: Utilizes GridSearchCV to find the best hyperparameters for each model.
- **Model Evaluation**: Evaluates models using various metrics such as confusion matrix, classification report, ROC AUC score, and ROC curve.
- **Feature Importance**: Visualizes feature importance to understand which features contribute most to the predictions.
- **Model Saving and Loading**: Saves the best-performing models and provides functionality to reload them.

## Files

- `heart_disease_prediction.py`: Contains the main code for data preprocessing, model training, evaluation, and saving/loading models.
- `README.md`: Provides an overview of the project and instructions.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Requirements

To run the code, you need to install the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `xgboost`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
