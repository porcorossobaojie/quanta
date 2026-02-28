# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:30:52 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Literal, Any
MODULE_DIR = __name__.split('.')[-2]

from .main import main as db
setattr(pd, MODULE_DIR, db())

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj: pd.DataFrame = pandas_obj

    def write(
        self,
        if_exists: Literal['fail', 'replace', 'append'] = 'append',
        index: bool = False,
        log: bool = True,
        **kwargs: Any
    ):
        db.write(self._obj, if_exists=if_exists, index=index, log=log, **kwargs)
