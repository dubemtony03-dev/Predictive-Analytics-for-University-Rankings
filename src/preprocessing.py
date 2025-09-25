import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

def clean_guardian(df, features):
    for col in features:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
    imputer = IterativeImputer(estimator=HistGradientBoostingRegressor(random_state=42))
    df[features] = imputer.fit_transform(df[features])
    return df
