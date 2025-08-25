# src/main.py
from data.preprocessing import clean_missing_values, encode_features, scale_features
from models.model import train_model, save_model, load_model