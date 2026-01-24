import pandas as pd
from .types import PredictionFrame

def robust_z_score(df: pd.DataFrame, value_col: str, pred: PredictionFrame) -> pd.Series:
    # residual / (MAD * 1.4826) to approximate std if normal
    resid = df[value_col] - pred.expected
    denom = pred.scale * 1.4826
    return resid / denom
