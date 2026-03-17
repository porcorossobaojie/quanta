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
        ret = getattr(flow, portfolio_type)(cls.returns).tools.log().astype('float32')
        entrade = ret.f.tradestatus().notnull()
        bench = cls.bench(bench).tools.log().astype('float32')
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)[entrade].fillna(0)
        w = pd.tools.halflife(long_periods+short_periods, halflife)[np.newaxis, :]
        
        ret_mom = ret.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)
        w_mom = entrade.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)
        bench_mom = bench.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)
        
        x = ((ret_mom - bench_mom) / w_mom).shift(short_periods)
        x = x.f.tradestatus(long_periods, halflife)
        return x
        
                
