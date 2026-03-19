# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:17:07 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd

from quanta.config import settings
config = settings('trade').pipline.tonghua

class main():
    def __init__(self, broker):
        [setattr(self, i,j) for i,j in getattr(config, broker).items()]
        self.broker = broker

    def read(self, path):
        data_mapping = {
            'xls': [pd.read_csv, {'encoding': 'gbk', 'sep': '\t'}],
            'xlsx': [pd.read_excel, {}],
            'csv': [pd.read_csv, {}]
            }

        func = data_mapping.get(self.settle.dtype, None)
        if func is None:
            raise ValueError('Undefined date type...')
        else:
            if path.split('.')[-1] != self.settle.dtype:
                path = '.'.join([path, self.settle.dtype])
            x = func[0](path, **func[1])[list(self.settle.columns.values())]
            return x

    def write(self, df, path):
        data_mapping = {
            'xls': ['to_excel', {}],
            'xlsx': ['to_excel', {}],
            'csv': ['to_csv', {'encoding': 'gbk'}]
            }
        func = data_mapping.get(self.order.dtype, None)
        if func is None:
            raise ValueError('Undefined date type...')
        else:
            if path.split('.')[-1] != self.order.dtype:
                path = '.'.join([path, self.order.dtype])
                df.columns = list(self.order.columns.values())
                getattr(df, func[0])(path, **func[1])






