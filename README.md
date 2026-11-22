# Income Prediction with Machine Learning

A focused Python implementation for classifying income in the Adult Census Income dataset. This repository is the standalone code project; the broader course notebooks and the original reported comparison are in [ai_advanced_course](https://github.com/carloscesar182/ai_advanced_course).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project overview

The project prepares the provided training, validation, and test CSV files, then trains Gradient Boosting, Random Forest, and Naive Bayes classifiers. It includes a separate evaluation module and an SHAP explanation module.

## Dataset

The project uses the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), which contains demographic and socioeconomic attributes. The repository includes pre-split CSV files in `data/`.

## Run the training script

Run these commands from the repository root with Python 3.10 or later:

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Start model training:

```bash
python src/main.py
```

The current entry point preprocesses the splits and trains the three Scikit-learn models. It does not yet call the evaluation module or save a metrics report. The evaluation helper is in `src/model_evaluation.py`.

## Results and tests

The original course experiment reports comparative results and test metrics in [ai_advanced_course](https://github.com/carloscesar182/ai_advanced_course). Those are reported course results and should be reproduced with the same data split and configuration before being relied on.

This repository contains legacy `test_*.py` scripts, but some still import modules from an older project layout. The full test suite is not currently a reliable pass/fail check; update those imports and assertions before claiming that all tests pass.

## Responsible use and limitations

The dataset includes demographic attributes and reflects historical social conditions. Model metrics alone do not demonstrate fairness across groups or suitability for real-world decisions. This is an educational project, not a tool for hiring, lending, or other consequential decisions.

## Technologies

- Python 3.10+
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- H2O AutoML and SHAP (used in course experiments and explanation work)

## Project structure

```text
.
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   ├── automl_model.py
│   ├── custom_models.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── explain_model.py
│   ├── main.py
│   └── model_evaluation.py
├── test_*.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
