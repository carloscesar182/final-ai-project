from data_loader import load_data, load_test_data
from data_preprocessing import replace_missing_values, encode_categorical_features, scale_numerical_features, encode_target_variable
from data_preprocessing import replace_test_missing_values, encode_test_categorical_features, scale_test_numerical_features, encode_test_target_variable
from custom_models import train_gb, train_rf, train_nb
from model_evaluation import evaluate_model

# caminho dos arquivos
train_path = 'data/train.csv'
validation_path = 'data/validation.csv'
test_path = 'data/test.csv'

# carregar os dados
train, validation = load_data(train_path, validation_path)
test = load_test_data(test_path)

# limpeza dos dados
train = replace_missing_values(train)
validation = replace_missing_values(validation)
test = replace_test_missing_values(test)

# separar features e target
X_train, y_train = train.iloc[:, :14], train.iloc[:, 14]
X_val, y_val = validation.iloc[:, :14], validation.iloc[:, 14]
X_test, y_test = test.iloc[:, :14], test.iloc[:, 14]

# encode das colunas categóricas
X_train_encoded, X_val_encoded, label_encoders = encode_categorical_features(X_train, X_val)
X_test_encoded, test_label_encoders = encode_test_categorical_features(X_test)

# encode da coluna alvo
y_train_encoded, y_val_encoded, le_target = encode_target_variable(y_train, y_val)
y_test_encoded, le_target_test = encode_test_target_variable(y_test)

# normalização das colunas numéricas (scale)
X_train_scaled, X_val_scaled, sc = scale_numerical_features(X_train_encoded, X_val_encoded)
X_test_scaled, sc_test = scale_test_numerical_features(X_test_encoded)

# treinar os modelos
best_gb = train_gb(X_train_scaled, y_train_encoded)
best_rf = train_rf(X_train_scaled, y_train_encoded)
best_nb = train_nb(X_train_scaled, y_train_encoded)