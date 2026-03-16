# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:23:15 2026

@author: Porco Rosso
"""

from functools import lru_cache
import numpy as np
import pandas as pd

from quanta import flow

from quanta.factors.barra._base import main as meta
#from._base import main as meta

class main(meta):
    _model_name = 'use4'
    
    @classmethod
    @lru_cache(maxsize=4)
    def momentum(
        cls, 
        long_periods = 504, 
        short_periods = 21,
        halflife = 126,
        bench='full',
        portfolio_type = 'astock'

    ):
        bench = cls.bench(bench).tools.log()
        ret = getattr(flow, portfolio_type)(cls.ret).tools.log()
        ret_values = ret.values
        ret_bool = ret.notnull()
        w = 0.5 ** (np.arange(long_periods+short_periods)[::-1] / halflife)
        w = w / w.sum()
        w_matrix = w.repeat(ret.shape[1]).reshape(-1, ret.shape[1])
        
                
