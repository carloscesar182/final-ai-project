import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# Gradient Boosting
def train_gb(X_train_scaled, y_train_encoded):
    gb = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [1.0]
    }
    grid = GridSearchCV(
        gb, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=-1, 
        verbose=1
    )
    grid.fit(X_train_scaled, y_train_encoded)
    return grid.best_estimator_

# Random Forest
def train_rf(X_train_scaled, y_train_encoded):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [400],
        'max_depth': [17, 19, 21],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }
    grid = GridSearchCV(
        rf, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=-1, 
        verbose=1
    )
    grid.fit(X_train_scaled, y_train_encoded)
    return grid.best_estimator_

# Naive Bayes
def train_nb(X_train_scaled, y_train_encoded):
    nb = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9,1e-8,1e-7,1e-6]
    }
    grid = GridSearchCV(
        nb, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=-1, 
        verbose=1
    )
    grid.fit(X_train_scaled, y_train_encoded)
    return grid.best_estimator_