# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:19:29 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from .core import _rolls as rolls
MODULE_DIR = __name__.split('.')[-2]

@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main:
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def __call__(
        self,
        window: int,
        min_periods: Optional[int] = None
    ) -> rolls:

        x = rolls(self._obj.to_frame(), window, min_periods).iloc[:, 0]
        return x

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def __call__(
        self,
        window: int,
        min_periods: Optional[int] = None
    ) -> rolls:

        x = rolls(self._obj, window, min_periods)
        return x
