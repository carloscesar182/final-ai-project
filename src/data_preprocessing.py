import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# nulos, missing values e outliers
def replace_missing_values(df):
    """Substitui '?' por 'Not-informed'."""
    df.replace('?', 'Not-informed', inplace=True)
    return df

def replace_test_missing_values(df_test):
    """Substitui '?' por 'Not-informed'."""
    df_test.replace('?', 'Not-informed', inplace=True)
    return df_test

def detect_outliers(df, numeric_cols):
    """Detecta e remove outliers."""
    iqr_results = {}
    for col in numeric_cols:
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        mediana = df[col].median()
        outliers = df[(df[col] < limite_inf) | (df[col] > limite_sup)]
        iqr_results[col] = {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'limite_inf': limite_inf,
            'limite_sup': limite_sup,
            'mediana': mediana,
            'outliers': len(outliers)
        }
    return iqr_results
        
# categorização das features da base de treino e validação
def encode_categorical_features(X_train, X_val):
    """Label encoder para variáveis categóricas."""
    label_encoders = {}
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    
    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_val_encoded[col] = le.transform(X_val[col])
        label_encoders[col] = le
    return X_train_encoded, X_val_encoded, label_encoders

# categorização das features da base de teste
def encode_test_categorical_features(X_test):
    """Label encoder para variáveis categóricas da base de teste."""
    test_label_encoders = {}
    X_test_encoded = X_test.copy()
    
    for col in X_test.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_test_encoded[col] = le.fit_transform(X_test[col])
        test_label_encoders[col] = le
    
    return X_test_encoded, test_label_encoders

# dimensionamento das colunas numéricas das bases de treino e validação
def scale_numerical_features(X_train_encoded, X_val_encoded):
    """StandarScaler para normalização das variáveis numéricas"""
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train_encoded)
    X_val_scaled = sc.transform(X_val_encoded)
    return X_train_scaled, X_val_scaled, sc

# dimensionamento das colunas numéricas da base de teste
def scale_test_numerical_features(X_test_encoded):
    """StandarScaler para normalização das variáveis numéricas da base de testes"""
    sc_test = StandardScaler()
    X_test_scaled = sc_test.fit_transform(X_test_encoded)
    return X_test_scaled, sc_test

# categorização da coluna alvo das bases de treino e validação
def encode_target_variable(y_train, y_val):
    """Label encoder para a variável alvo "y"."""
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_val_encoded = le_target.transform(y_val)
    return y_train_encoded, y_val_encoded, le_target

# categorização da coluna alvo da base de teste
def encode_test_target_variable(y_test):
    """Label encoder para a variável alvo "y" da base de testes."""
    le_target_test = LabelEncoder()
    y_test_encoded = le_target_test.fit_transform(y_test)
    return y_test_encoded, le_target_test