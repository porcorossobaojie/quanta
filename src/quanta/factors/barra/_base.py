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
    def _beta(
        cls, 
        periods = 252, 
        halflife=None,
        bench = 'full',
        portfolio_type = 'astock'
    ):
        halflife = periods//4 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.returns).fillna(0)
        entrade = ret.f.tradestatus().notnull()
        bench = cls.bench(bench)
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)[entrade].fillna(0)
        w = pd.tools.halflife(periods, halflife)[np.newaxis, :]
        
        roll = pd.concat({'ret':ret, 'w':entrade, 'bench':bench}, axis=1).rolling(periods).apply(lambda x: w @ x, raw=True)
        y = (ret - roll.ret /roll.w)
        x = (bench - roll.bench / roll.w)
        beta = (x * y).rolling(periods).apply(lambda x: w @ x, raw=True) / (x * x).rolling(periods).apply(lambda x: w @ x, raw=True)        
        alpha = roll.ret / roll.w - beta * roll.bench / roll.w
        resid = (ret - alpha.bfill() - beta.bfill() * bench) ** 2
        resid = resid.rolling(periods).apply(lambda x: w @ x, raw=True)
        resid = resid / roll.w
        df = pd.concat({i:j.f.tradestatus(periods, halflife) for i,j in {'alpha':alpha, 'beta':beta, 'resid':resid}}, axis=1)
        return df
    
    @classmethod
    def beta(cls, periods=252, bench='full', portfolio_type='astock'):
        return cls._beta(periods, bench)[1]
    
            
        

        
        
        
