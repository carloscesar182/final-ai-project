import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test, feature_names):
    """
    Explica previsões de um modelo de ML usando SHAP.
    
    Args:
        model: modelo treinado
        X_train: dados de treino (pré-processados)
        X_test: dados de teste (pré-processados)
        feature_names: lista de nomes das features
    """
    # inicializa o explainer
    explainer = shap.Explainer(model, X_train)
    
    # calcula valores do shap para o teste
    shapel_values = explainer(X_test)
    
    # plota o resumo global das features
    shap.summary_plot(shapel_values, X_test, feature_names=feature_names)
    
    # plot os valores de shap para cada instância
    shap.plots.waterfall(shapel_values[0])