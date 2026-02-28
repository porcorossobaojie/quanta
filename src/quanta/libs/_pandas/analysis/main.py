# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 17:59:34 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from .core import *
MODULE_DIR = __name__.split('.')[-2]

@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def maxdown(
        self,
        iscumprod: bool = True
    ):
        series = maxdown(self._obj.to_frame(), iscumprod=iscumprod)
        return series

    def sharpe(
        self,
        iscumprod: bool = False,
        periods: int = 252
    ):
        series = sharpe(self._obj.to_frame(), iscumprod, periods)
        return series

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def maxdown(
        self,
        iscumprod: bool = False
    ):
        df = maxdown(self._obj, iscumprod=iscumprod)
        return df

    def sharpe(
        self,
        iscumprod: bool = False,
        periods: int = 252
    ) -> pd.DataFrame:
        df = sharpe(self._obj, iscumprod, periods)
        return df

    def effective(self):
        df = effective(self._obj)
        return df
