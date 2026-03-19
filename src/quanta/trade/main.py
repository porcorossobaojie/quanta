# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:42:22 2026

@author: Porco Rosso
"""

from pathlib import Path

import pandas as pd

from quanta import flow
from quanta.trade import pipline
from quanta.config import settings

char = settings('trade').strategy_001.BJ_13611823855
config = settings('data').public_keys.recommand_settings 

class main:
    path = settings('trade').system

    def __init__(self, **kwargs):
        [setattr(self, f"_{i}", j) for i,j in kwargs.items()]
        self.__env_init__()
        
    @property
    def pipline(self):
        x = getattr(pipline, self._pipline)
        x = x(broker=self._broker)
        return x
    
    @property
    def portfolio_type(self):
        x = getattr(self, '_portfolio_type', config.key.astock_code)
        return x
    
    @property
    def __order_path__(self):
        x = Path(self.path) / self._strategy / self._name / self.path.order
        return x
    
    @property
    def __settle_path__(self):
        x = Path(self.path) / self._strategy / self._name / self.path.order
        return x

    def __env_init__(self):
        self.__order_path__.mkdir(parents=True, exist_ok=True)
        self.__settle_path__.mkdir(parents=True, exist_ok=True)
        
    def __get_files_name__(self, path):
        files = [f.name for f in path.iterdir() if f.is_file()]
        return files        

    def settle(self, file_name_by_date):
        x = self.pipline.read(self.__settle_path__ / file_name_by_date)
        x = x.set_index(x.columns[0])
        x = x.iloc[:, 0]
        x.index.name = self.portfolio_type
        x.name = pd.to_datetime(file_name_by_date) + pd.Timedelta(config.key.time_bias)
        
