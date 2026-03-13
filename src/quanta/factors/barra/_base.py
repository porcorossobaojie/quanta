# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:31:33 2026

@author: Porco Rosso
"""
from functools import lru_cache
import numpy as np
import pandas as pd

from quanta import flow
from quanta.factors._base.main import main as meta
from quanta.config import settings
config = settings('factors')

class main(meta):
    finance = config.finance_keys
    
    @classmethod
    @lru_cache(maxsize=1)
    def size(cls):
        x = (flow.astock(cls.finance.val_mv) * 1e8).tools.log()
        return x
    
    @classmethod
    @lru_cache(maxsize=1)
    def bm(cls):
        x = flow.astock(cls.finance.pb) ** -1
        return x        
    
    @classmethod
    @lru_cache(maxsize=1)
    def non_size(cls):
        df = cls.size()
        df = (df ** 3).stats.neutral(me=df, weight= (df ** 0.5).values).resid.tools.log().stats.standard()
        return df
    
    @classmethod
    @lru_cache(maxsize=8)
    def beta(cls, periods=252, bench='full'):
        bench = cls.bench(bench)
        ret = flow.astock(cls.returns).f.filtered()
        bench = pd.DataFrame(
            bench.values.repeat(ret.shape[-1]).reshape(ret.shape[0], -1), 
            index = ret.index, 
            columns = ret.columns
        )
        w = pd.tools.half_life(periods, periods//4)
        
        vol_bench = bench - bench.rolling(periods, periods//4).mean()
        vol_ret = ret -  ret.rolling(periods, periods//4).mean()
        vol_up = vol_bench * vol_ret
        vol_up = pd.tools.array_roll(vol_up.values, periods)
        vol_ret = pd.tools.array_roll(vol_ret.values ** 2, periods)
        vol_1 = []
        vol_2 = []
        for i in range(vol_up.shape[0]):
            vol_1.append(np.nansum(vol_up[i] * w[:, np.newaxis], axis=0))
            vol_2.append(np.nansum(vol_ret[i] * w[:, np.newaxis], axis=0))
        vol_1 = np.array(vol_1)
        vol_2 = np.array(vol_2)
        vol_1 = np.where(vol_1 != 0, vol_1, np.nan)
        vol_2 = np.where(vol_2 != 0, vol_2, np.nan)
        x = pd.DataFrame(vol_1 / vol_2, index=ret.index[periods-1:], columns=ret.columns)


        
        
        
