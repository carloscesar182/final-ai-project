# Income Prediction with Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## About the project

This course capstone project implements an end-to-end Machine Learning pipeline to predict income levels from demographic and socioeconomic features in the Adult (Census Income) dataset.

The goal was to build an organized, reproducible, and explainable solution covering the process from data loading to model training, comparison, and interpretation.

## Key features

- Loads training, validation, and test datasets
- Handles missing values
- Encodes categorical features
- Scales numerical features
- Trains Gradient Boosting, Random Forest, and Naive Bayes models
- Evaluates models using classification metrics
- Includes a structure prepared for model interpretability with SHAP
- Provides automated tests for preprocessing, training, evaluation, and explainability

## Technologies

- Python 3.10+
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- H2O AutoML
- SHAP
- Pytest

## Project structure

```text
.
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── custom_models.py
│   ├── automl_model.py
│   └── model_evaluation.py
```
