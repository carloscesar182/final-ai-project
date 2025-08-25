# Final AI Project 🎯

Projeto final do curso de Inteligência Artificial, desenvolvido em Python.  
O objetivo é construir um pipeline completo de **Machine Learning**, incluindo:
- Pré-processamento de dados
- Treinamento de modelos customizados
- AutoML
- Avaliação de modelos
- Interpretação de modelos (XAI)

---

## 📂 Estrutura do Projeto

final_ai_project/
│── src/
│ ├── data_loader.py # Funções para carregar datasets
│ ├── data_preprocessing.py # Funções de limpeza, transformação e split dos dados
│ ├── custom_models.py # Definição de modelos customizados (ex: GradientBoostingClassifier)
│ ├── automl_model.py # Pipeline de AutoML (AutoSklearn, Optuna, etc.)
│ ├── model_evaluation.py # Funções para métricas e avaliação dos modelos
│ ├── explain_model.py # Interpretação dos modelos (SHAP, LIME, etc.)
│ └── main.py # Script principal para rodar o pipeline completo
│
│── tests/
│ ├── test_preprocessing.py # Testes unitários do pré-processamento
│ ├── test_train_models.py # Testes de treinamento de modelos
│ ├── test_evaluation.py # Testes de avaliação
│ └── test_xai.py # Testes de explicabilidade
│
│── requirements.txt # Dependências do projeto
│── README.md # Documentação do projeto

---

## ⚙️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/carloscesar182/final-ai-project.git
cd final-ai-project
```

2. Crie o ambiente virtual:
```bash
python -m venv .venv
```

3. Ative o ambiente:
- Windows:

```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. Instale as dependências:

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar
Rode o pipeline completo com:

```bash
python src/main.py
```

## 🧪 Testes
Para rodar os testes unitários:

```bash
pytest
```

## 📊 Exemplo de Uso

Um exemplo simples de como rodar o pipeline no Python:
```python
from src.data_loader import load_dataset
from src.data_preprocessing import preprocess_data
from src.custom_models import train_custom_model
from src.model_evaluation import evaluate_model
from src.explain_model import explain_with_shap

# 3. Carregar dados (ex: dataset Titanic, Iris, etc.)
df = load_dataset("iris")

# 2. Pré-processar
X_train, X_test, y_train, y_test = preprocess_data(df, target="species")

# 3. Treinar modelo
model = train_custom_model(X_train, y_train)

# 4. Avaliar
metrics = evaluate_model(model, X_test, y_test)
print("Resultados:", metrics)

# 5. Interpretar
explain_with_shap(model, X_test)
```

## 📊 Modelos
Atualmente, o projeto inclui:
- GradientBoostingClassifier (modelo base)
- AutoML para seleção e tuning de outros modelos
- Interpretação via SHAP/LIME

## 🛠 Tecnologias
- Python 3.10+
- Scikit-learn
- Pandas & Numpy
- SHAP / LIME (XAI)
- Pytest (testes)
- Auto-sklearn / Optuna (AutoML)

## ✨ Próximos Passos
- Adicionar novos modelos customizados
- Melhorar explicabilidade com gráficos interativos
- Deploy em API/Streamlit