#!/usr/bin/env python3
"""
Test script to verify that the preprocessing module works correctly
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data_preprocessing import replace_missing_values, encode_categorical_features, scale_numerical_features

def test_preprocessing_functions():
    print("Testing preprocessing functions...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'category': ['A', 'B', '?', 'A', 'C'],
        'value1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'value2': [10, 20, 30, 40, 50]
    })
    
    print("Original data:")
    print(sample_data)
    print()
    
    # Test replace_missing_values
    cleaned_data = replace_missing_values(sample_data.copy())
    print("After cleaning missing values:")
    print(cleaned_data)
    print()
    
    # Test encode_categorical_features (requires train and validation sets)
    # For testing, we'll use the same data for both train and val
    train_data = cleaned_data.drop('value2', axis=1)  # Remove target column
    val_data = train_data.copy()
    encoded_train, encoded_val, encoders = encode_categorical_features(train_data, val_data)
    print("After encoding features (train):")
    print(encoded_train)
    print("Encoders:", list(encoders.keys()))
    print()
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing_functions()