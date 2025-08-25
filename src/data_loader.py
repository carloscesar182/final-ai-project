import pandas as pd

def load_data(train_path, validation_path):
    """Carrega os dados de treino e validação do arquivo CSV."""
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    return train, validation

def load_test_data(test_path):
    """Carrega os dados de teste do arquivo CSV."""
    test = pd.read_csv(test_path)
    return test