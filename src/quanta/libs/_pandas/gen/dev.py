# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:49:40 2026

@author: admin
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union

from ..tools.core import fillna as fillna_func

def portfolio(df_obj: pd.DataFrame,
    returns: pd.DataFrame,
    weight: Optional[pd.DataFrame] = None,
    shift: int = 1,
    roll: int = 1,
    fillna: bool = False
):
    returns = returns.sort_index().rolling(roll).mean() if roll > 1 else returns.sort_index()
    df_obj = (fillna_func(df_obj.sort_index(), returns.index) if fillna else df_obj).shift(shift)

    x = pd.get_dummies(df_obj, prefix_sep='_SEP_KEY_')
    columns = pd.MultiIndex.from_tuples([tuple(i) for i in x.columns.str.split('_SEP_KEY_')], names=[df_obj.columns.name, 'portfolio'])
    x.columns = columns
    ret = returns.reindex(x.columns.get_level_values(0), axis=1)
    ret.columns = columns
    ret = ret[x]
    if weight is None:
        obj = ret.groupby('portfolio', axis=1).mean()
    else:
        weight = (fillna_func(weight, returns.index) if fillna else weight)
        weight = weight.reindex(x.columns.get_level_values(0), axis=1)
        weight.columns = columns
        weight = weight[x]
        obj = (ret * weight).groupby('portfolio', axis=1).sum(min_count=1) / weight.groupby('portfolio', axis=1).sum(min_count=1)
    return obj
