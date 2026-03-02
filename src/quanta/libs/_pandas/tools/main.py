# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:52:33 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, List, Any
from .core import *
from quanta.libs.utils._decorator import doc_inherit

MODULE_DIR = __name__.split('.')[-2]


@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas DataFrame accessor for utility tools, including data filling,
    shifting, logarithmic transformations, and array rolling.
    ---------------------------------------------------------------------------
    用于实用工具的 Pandas DataFrame 访问器, 包括数据填充, 移动, 对数变换和数组
    滚动.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @doc_inherit(fillna)
    def fillna(
        self,
        fill_list: List[Any]
    ) -> pd.DataFrame:
        return fillna(self._obj, fill_list)

    @doc_inherit(shift)
    def shift(
        self,
        n: int = 1
    ) -> pd.DataFrame:
        return shift(self._obj, n)

    @doc_inherit(log)
    def log(
        self,
        bias_adj: Union[int, float] = 1,
        abs_adj: bool = True
    ) -> pd.DataFrame:
        return log(self._obj, bias_adj, abs_adj)

    @doc_inherit(array_roll)
    def array_roll(
        self,
        periods: int
    ) -> np.ndarray:
        return array_roll(self._obj.values, periods)


@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas Series accessor for utility tools, specifically logarithmic
    transformations.
    ---------------------------------------------------------------------------
    用于实用工具的 Pandas Series 访问器, 特指对数变换.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    @doc_inherit(log)
    def log(
        self,
        bias_adj: Union[int, float] = 1,
        abs_adj: bool = True
    ) -> pd.Series:
        return log(self._obj, bias_adj, abs_adj)


class class_obj:
    """
    ===========================================================================
    Static utility class registered to Pandas for standalone tool access.
    ---------------------------------------------------------------------------
    注册到 Pandas 的静态工具类, 用于独立工具访问.
    ---------------------------------------------------------------------------
    """
    half_life = staticmethod(half_life)
    array_roll = staticmethod(array_roll)


setattr(pd, MODULE_DIR, class_obj)
