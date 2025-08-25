import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(model, X_test_scaled, y_test_encoded, le_target=None, plot_confusion=True, title="Matriz de Confusão"):
    y_test_pred = model.predict(X_test_scaled)
    metrics = {
        'f1': f1_score(y_test_encoded, y_test_pred),
        'accuracy': accuracy_score(y_test_encoded, y_test_pred),
        'precision': precision_score(y_test_encoded, y_test_pred),
        'recall': recall_score(y_test_encoded, y_test_pred),
        'report': classification_report(y_test_encoded, y_test_pred),
        'confusion_matrix': confusion_matrix(y_test_encoded, y_test_pred)
    }
    
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Classification Report: \n{metrics['report']}")
    
    if plot_confusion:
        plt.figure(figsize=(6, 4))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=le_target.classes_ if le_target else ['0', '1'],
                    yticklabels=le_target.classes_ if le_target else ['0', '1'])
        plt.xlabel('Valores Preditos')
        plt.ylabel('Valores Reais')
        plt.title(title)
        plt.show()
    
    return metrics