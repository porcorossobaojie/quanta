# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:49:44 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, Dict
from .core import *
from quanta.libs.utils._decorator import doc_inherit

MODULE_DIR = __name__.split('.')[-2]


@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas DataFrame accessor for generation and grouping operations,
    including ranking, binning, and portfolio construction.
    ---------------------------------------------------------------------------
    用于生成和分组操作的 Pandas DataFrame 访问器, 包括排名, 分箱和投资组合构建.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @doc_inherit(group)
    def group(
        self,
        rule: Union[Dict, List] = np.linspace(0, 1, 11).round(2).tolist(),
        pct: bool = True,
        order: bool = False,
        nlevels: Optional[List[Union[int, str]]] = None,
    ) -> pd.DataFrame:
        df: pd.DataFrame = group(self._obj, rule=rule, pct=pct, order=order, nlevels=nlevels)
        return df

    @doc_inherit(weight)
    def weight(
        self,
        w_df: Optional[pd.DataFrame] = None,
        fillna: bool = True,
        pct: bool = True,
    ) -> pd.DataFrame:
        return weight(self._obj, w_df=w_df, fillna=fillna, pct=pct)

    @doc_inherit(portfolio)
    def portfolio(
        self,
        returns: pd.DataFrame,
        weight: Optional[pd.DataFrame] = None,
        shift: int = 1,
        roll: int = 1,
        fillna: bool = True
    ) -> pd.DataFrame:
        return portfolio(self._obj, returns=returns, weight=weight, shift=shift, roll=roll, fillna=fillna)

    @doc_inherit(cut)
    def cut(
        self,
        right: Union[int, float],
        rng_right: Union[int, float] = 0,
        left: Union[int, float] = 0,
        rng_left: Union[int, float] = 0,
        pct: bool = False,
        ascending: bool = False
    ) -> pd.DataFrame:
        return cut(self._obj, left, right, rng_left, rng_right, pct, ascending)

    @doc_inherit(part_cut)
    def part_cut(
        self,
        right: Union[int, float],
        rng_right: Union[int, float] = 0,
        left: Union[int, float] = 0,
        rng_left: Union[int, float] = 0,
        pct: bool = False,
        ascending: bool = False,
        part: int = 5
    ) -> pd.DataFrame:
        return part_cut(self._obj, left, right, rng_left, rng_right, pct, ascending, part)
