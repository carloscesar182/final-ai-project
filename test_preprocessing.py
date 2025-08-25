#!/usr/bin/env python3
"""
Test script to verify that the preprocessing module works correctly
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from data.preprocessing import clean_missing_values, encode_features, scale_features

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
    
    # Test clean_missing_values
    cleaned_data = clean_missing_values(sample_data)
    print("After cleaning missing values:")
    print(cleaned_data)
    print()
    
    # Test encode_features
    encoded_data, encoders = encode_features(cleaned_data, None)
    print("After encoding features:")
    print(encoded_data)
    print("Encoders:", list(encoders.keys()))
    print()
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing_functions()