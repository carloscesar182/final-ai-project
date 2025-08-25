# src/data/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui '?' por 'Not-informed'."""
    return df.replace('?', 'Not-informed')

def encode_features(df: pd.DataFrame, encoders=None):
    """Label encoder para variáveis categóricas."""
    encoders = encoders or {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = encoders.get(col, LabelEncoder())
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    return df_encoded, encoders

def scale_features(tran_df, val_df):
    """StandarScaler para normalização das variáveis numéricas"""
    scaler = StandardScaler()
    return scaler.fit_transform(tran_df), scaler.transform(val_df)