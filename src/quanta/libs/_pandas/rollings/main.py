# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:19:29 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from .core import _rolls as rolls
from quanta.libs.utils._decorator import doc_inherit

MODULE_DIR = __name__.split('.')[-2]


@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main:
    """
    ===========================================================================
    Pandas Series accessor for rolling window operations.
    ---------------------------------------------------------------------------
    用于滚动窗口操作的 Pandas Series 访问器.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def __call__(
        self,
        window: int,
        min_periods: Optional[int] = None
    ) -> rolls:
        """
        =======================================================================
        Initializes a rolling window object for the Series.

        Parameters
        ----------
        window : int
            The size of the rolling window.
        min_periods : Optional[int]
            Minimum number of observations in window required to have a value.

        Returns
        -------
        rolls
            A rolling window operation container.
        -----------------------------------------------------------------------
        为 Series 初始化滚动窗口对象.

        参数
        ----
        window : int
            滚动窗口的大小.
        min_periods : Optional[int]
            窗口中需要有值的最小观测数.

        返回
        ----
        rolls
            滚动窗口操作容器.
        -----------------------------------------------------------------------
        """
        x = rolls(self._obj.to_frame(), window, min_periods)
        return x


@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas DataFrame accessor for rolling window operations.
    ---------------------------------------------------------------------------
    用于滚动窗口操作的 Pandas DataFrame 访问器.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def __call__(
        self,
        window: int,
        min_periods: Optional[int] = None
    ) -> rolls:
        """
        =======================================================================
        Initializes a rolling window object for the DataFrame.

        Parameters
        ----------
        window : int
            The size of the rolling window.
        min_periods : Optional[int]
            Minimum number of observations in window required to have a value.

        Returns
        -------
        rolls
            A rolling window operation container.
        -----------------------------------------------------------------------
        为 DataFrame 初始化滚动窗口对象.

        参数
        ----
        window : int
            滚动窗口的大小.
        min_periods : Optional[int]
            窗口中需要有值的最小观测数.

        返回
        ----
        rolls
            滚动窗口操作容器.
        -----------------------------------------------------------------------
        """
        x = rolls(self._obj, window, min_periods)
        return x
