import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from typing import cast, Any
import numpy as np


def train_automl(X_train_scaled: Any, y_train_encoded: Any, X_val_scaled: Any, y_val_encoded: Any, max_runtime_secs: int = 300) -> H2OAutoML:
    h2o.init()
    
    # converter numpy para pandas
    feature_cols = [f'feature_{i}' for i in range(X_train_scaled.shape[1])]
    X_train_df = pd.DataFrame(X_train_scaled, columns=cast(Any, feature_cols))
    y_train_series: pd.Series = pd.Series(y_train_encoded, name='target')
    
    val_feature_cols = [f'feature_{i}' for i in range(X_val_scaled.shape[1])]
    X_val_df = pd.DataFrame(X_val_scaled, columns=cast(Any, val_feature_cols))
    y_val_series: pd.Series = pd.Series(y_val_encoded, name='target')
    
    train_df = H2OFrame(pd.concat([X_train_df, y_train_series], axis=1))
    valid_df: H2OFrame = H2OFrame(pd.concat([X_val_df, y_val_series], axis=1))
    
    # Converter a coluna alvo para fator para classificação
    train_df['target'] = cast(Any, train_df['target']).asfactor()
    valid_df['target'] = cast(Any, valid_df['target']).asfactor()
    
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        seed=42,
        nfolds=0,
        sort_metric='f1'
    )
    aml.train(
        x=train_df.columns[:-1],
        y='target',
        training_frame=train_df,
        validation_frame=valid_df
    )
    return aml