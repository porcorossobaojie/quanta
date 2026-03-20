# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:36:01 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from functools import lru_cache

from quanta import flow, faclib
from quanta.config import settings
char = settings('trade').strategy_001.BJ_13611823855
from quanta.trade import account


class main:
    
    @classmethod
    def _internal_data(cls):
        pass
    
    @classmethod
    def factor(cls):
        return faclib.barra.us4.bm()
    
    def __init__(self, account_obj):
        self.account = account_obj
    
    @lru_cache(maxsize=4)
    def pool(self, index_members=None):
        x = self.factor().f.filtered()
        if index_members is not None:
            x = x.f.index_members(index_members)
        x = x.iloc[-1].dropna().sort_index()
        x = pd.f.Series(x)
        return x
    
    def ranker(self, lst=None, index_members=None):
        x = self.pool(index_members).rank(ascending=False)
        if lst is not None:
            x = x.loc[lst]
        return x
    
    def settle(self):
        x = self.account.settle()
        x = x[x > 0]
        return x
    
    def ensell(self, high_limit=True):
        x = self.settle()
        sells = x[self.ranker(lst = x.index) > (self.account._portfolio_count + self.account._portfolio_range)]
        if high_limit:
            sells = sells[~sells.f.info('tradestatus').astype('bool') & (sells.f.info('close') < sells.f.info('high_limit'))]
        return sells
    
    def hold(self):
        x = self.settle()
        x = x[x.index.difference(self.ensell().index)]
        return x
    
    def enbuy(self, high_limit=True, low_limit=True, index_member=None):
        rank = self.ranker().sort_values()
        if high_limit:
            rank = rank[~rank.f.info('tradestatus').astype('bool') & (rank.f.info('close') < rank.f.info('high_limit'))]
        if low_limit:
            rank = rank[~rank.f.info('tradestatus').astype('bool') & (rank.f.info('close') > rank.f.info('low_limit'))]
        rank = rank[rank.index.difference(self.hold().index)].nsmallest(self.ensell().shape[0])
        return rank
    
    def rebalance(self, **kwargs):
        pass
        
    
self = main(account(**char))
        
        
        
        
        
        

            
