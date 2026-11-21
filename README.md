# Previsão de Renda com Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Sobre o projeto

Este projeto de conclusão de curso implementa um pipeline completo de Machine Learning para prever a faixa de renda de pessoas a partir de características demográficas e socioeconômicas do dataset Adult (Census Income).

O objetivo foi desenvolver uma solução organizada, reproduzível e explicável, abrangendo desde o carregamento dos dados até o treinamento, comparação e interpretação dos modelos.

## Principais funcionalidades

- Carregamento dos conjuntos de treino, validação e teste
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização de variáveis numéricas
- Treinamento de Gradient Boosting, Random Forest e Naive Bayes
- Avaliação dos modelos com métricas de classificação
- Estrutura preparada para interpretabilidade com SHAP
- Testes automatizados para pré-processamento, treinamento, avaliação e explicabilidade

## Tecnologias

- Python 3.10+
- Pandas e NumPy
- Scikit-learn
- Matplotlib e Seaborn
- H2O AutoML
- SHAP
- Pytest

## Estrutura do projeto

```text
.
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── custom_models.py
│   ├── automl_model.py
│   ├── model_evaluation.py
│   ├── explain_model.py
│   └── main.py
├── test_evaluation.py
├── test_preprocessing.py
├── test_train_models.py
├── test_xai.py
├── requirements.txt
└── README.md
```

## Instalação

```bash
git clone https://github.com/carloscesar182/final-ai-project.git
cd final-ai-project
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como executar

A partir da raiz do projeto:

```bash
python src/main.py
```

Os dados de entrada devem permanecer em `data/train.csv`, `data/validation.csv` e `data/test.csv`.

## Testes

```bash
pytest
```

## Sobre o dataset

Este projeto utiliza a base pública Adult Income, também conhecida como Census Income. Cada registro representa uma pessoa e contém atributos como idade, escolaridade, ocupação, horas trabalhadas e situação civil.

A variável-alvo indica se a renda anual ultrapassa um determinado limite. O dataset pode apresentar desbalanceamento entre as classes e exigia tratamento cuidadoso dos dados antes do treinamento.

## Próximas melhorias

- Publicar métricas finais de forma comparativa
- Adicionar matriz de confusão e visualizações SHAP
- Criar uma pipeline única com Scikit-learn
- Adicionar validação cruzada e busca de hiperparâmetros
- Disponibilizar o modelo por meio de uma API ou aplicação web
- Completar uma execução ponta a ponta com instruções de reprodução

## Licença

Este projeto está licenciado sob a licença MIT.

## Contato

- **Autor:** Carlos Ferreira
- **LinkedIn:** https://www.linkedin.com/in/carloscferreira/
- **E-mail:** carloscesar182@gmail.com
