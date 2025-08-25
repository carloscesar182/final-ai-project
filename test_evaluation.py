#!/usr/bin/env python3
"""
Test script to verify that the evaluation module works correctly
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from evaluation.evaluate_models import evaluate_model

def test_evaluation_functions():
    print("Testing evaluation functions...")
    
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Sample data created and model trained.")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print()
    
    # Test evaluate_model function (without plotting to avoid display issues)
    print("Testing evaluate_model function...")
    metrics = evaluate_model(model, X_val, y_val, label_encoder=None, plot_confusion=False)
    
    print("\nEvaluation completed successfully!")
    print("Available metrics:", list(metrics.keys()))
    print()
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_evaluation_functions()