# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:40 2026

@author: Porco Rosso
"""
import pandas as pd
from functools import lru_cache
#from .base import Series, DataFrame
from quanta.libs._flow._cap.base import Series, DataFrame

class Unit():
    def __init__(self, **kwargs):
        series_instance = Series._config.recommand_settings.to_dict()
        series_instance = series_instance | {'state': 'settle', 'unit':'share'} | {i: j for i,j in kwargs.items() if i in series_instance.keys()}
        self._meta_data = {'settle': Series([], **series_instance)} | {i: j for i,j in kwargs.items() if i not in series_instance.keys()}
        self._exist_info = list(self._meta_data.keys())
        [setattr(self, i, j) for i,j in self._meta_data.items()]
        
    def __get_state__(self, *order_chain, method):
        states = [getattr(self, i, None) for i in order_chain]
        state = next((c for c in states if c is not None), None)
        if state is None:
            raise AttributeError(f"Failed to get attribute: <{order_chain[0]}>")
        return getattr(state, method)()
    
    @property
    def signal(self):
        chain = ['_signal', '_order', '_trade']
        return self.__get_state__(*chain, method='signal')
    @signal.setter
    def signal(self, v):
        if not isinstance(v, Series):
            v = Series(v)
        setattr(self, '_signal', v)
    
    @property
    def order(self):
        chain = ['_order', '_signal', '_trade']
        return self.__get_state__(*chain, method='order')
        
    @property
    def trade(self):
        chain = ['_trade', '_order', '_signal']
        return self.__get_state__(*chain, method='trade')
    
    @property
    def entrade(self):
        x = self.trade
        x = x.enbuy() + x.ensell()
        return x
    
    @property
    def settle(self):
        x = self._settle
        if x.name is None:
            x.name = self.signal.name
        if x.index.name is None:
            x.index.name = self.signal.index.name
        return x
    @settle.setter
    def settle(self, v):
        if not isinstance(v, Series):
            v = Series(v)
        setattr(self, '_settle', v)
        
    @property
    def trade_cost(self):
        return self.entrade.trade_cost(trade_check=False)
    
    def forward(self, trade_cost=True):
        settle = self.settle.share() + self.entrade.share()
        if trade_cost:
            settle.cash = settle.cash + self.trade_cost
        return settle
    
    def target_position(self, target):
        if not isinstance(target, Series):
            target = Series(target)
        target = target.assets(self.forward().total_assets())
        signal = (target - self.forward().assets()).share().signal()
        return signal

        
        
        

    
'''
from quanta import flow
ret = flow.astock('ret')
series = Series(ret.iloc[200, :200]).abs().share(1000)
self = Unit(signal=series, cash=1000)
target = ret.iloc[201, 100:300].abs()
target = Series(target).weight()
'''
