# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:22:04 2026

@author: Porco Rosso
"""
import numpy as np
import pandas as pd

__all__ = ['maxdown', 'sharpe', 'effective']

def maxdown(df_obj, iscumprod: bool = True):
    if not iscumprod:
        x = df_obj.add(1, fill_value=0).cumprod()
        x[df_obj.isnull()] = pd.NA
    else:
        x = df_obj

    max_flow = x.expanding(min_periods=1).max()
    down_date = (x / max_flow).idxmin()
    max_down_series = x / max_flow - 1

    down_info = [(x.loc[j, i] if pd.notnull(j) else np.nan, max_down_series.loc[j, i] if pd.notnull(j) else np.nan) for i, j in down_date.items()]
    down_value, max_down_value = list(zip(*down_info))

    up_value = pd.Series([max_flow.loc[j, i] if pd.notnull(j) else np.nan for i, j in down_date.items()], index=down_date.index)
    up_date = max_flow[max_flow == up_value].idxmin()

    df = pd.DataFrame(
        [up_date.values, up_value.values, down_date.values, down_value, max_down_value],
        columns=df_obj.columns,
        index=['Maxdown_Start_Date', 'Maxdown_Start_Value', 'Maxdown_End_Date', 'Maxdown_End_Value', 'Maxdwon_Percent']
    )
    return df

def sharpe(df_obj, iscumprod: bool=True, periods=252):
    x = df_obj if not iscumprod else df_obj.pct_change(fill_method=None)
    y = x.mean() / x.std()

    if periods is not None:
        periods = min(len(x), periods)
        y = y * (periods ** 0.5)

    y.name = f'sharpe_ratio(periods={periods})'
    return y

def effective(df_obj):
    x = df_obj.diff(axis=1)
    x = np.sign(x) * x ** 2
    x = x.sum(axis=1)
    return x