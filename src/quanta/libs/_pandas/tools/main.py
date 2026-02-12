# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:52:33 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from .core import *
MODULE_DIR = __name__.split('.')[-2]

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def fillna(
        self,
        fill_list
    ):
        return fillna(self._obj, fill_lists)

    def shift(
        self,
        n: int = 1
    ):
        return shift(self._obj, n)

    def log(
        self,
        bias_adj: float = 1,
        abs_adj: bool = True
    ) -> pd.DataFrame:

        return log(self._obj, bias_adj, abs_adj)

    def array_roll(
        self,
        periods
    ) -> np.ndarray:
        return array_roll(self.values, periods)

@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def log(
        self,
        bias_adj: float = 1,
        abs_adj: bool = True
    ) -> pd.DataFrame:
        return log(self._obj, bias_adj, abs_adj)

class class_obj:
    half_life = half_life
    array_roll = array_roll

setattr(pd, MODULE_DIR, class_obj)
