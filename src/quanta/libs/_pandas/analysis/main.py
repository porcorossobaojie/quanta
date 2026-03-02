# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 17:59:34 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from .core import *
from quanta.libs.utils._decorator import doc_inherit

MODULE_DIR = __name__.split('.')[-2]


@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas Series accessor for quantitative analysis operations.
    ---------------------------------------------------------------------------
    用于量化分析操作的 Pandas Series 访问器.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    @doc_inherit(maxdown)
    def maxdown(
        self,
        iscumprod: bool = True
    ) -> pd.DataFrame:
        series = maxdown(self._obj.to_frame(), iscumprod=iscumprod)
        return series

    @doc_inherit(sharpe)
    def sharpe(
        self,
        iscumprod: bool = False,
        periods: int = 252
    ) -> pd.Series:
        series = sharpe(self._obj.to_frame(), iscumprod, periods)
        return series


@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas DataFrame accessor for quantitative analysis operations.
    ---------------------------------------------------------------------------
    用于量化分析操作的 Pandas DataFrame 访问器.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @doc_inherit(maxdown)
    def maxdown(
        self,
        iscumprod: bool = False
    ) -> pd.DataFrame:
        df = maxdown(self._obj, iscumprod=iscumprod)
        return df

    @doc_inherit(sharpe)
    def sharpe(
        self,
        iscumprod: bool = False,
        periods: int = 252
    ) -> pd.Series:
        df = sharpe(self._obj, iscumprod, periods)
        return df

    @doc_inherit(effective)
    def effective(self) -> pd.Series:
        df = effective(self._obj)
        return df
