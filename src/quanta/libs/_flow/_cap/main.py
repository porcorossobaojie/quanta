# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:11:56 2026

@author: Porco Rosso
"""
from functools import lru_cache
import numpy as np
import pandas as pd

from quanta.libs import _pandas
from quanta.libs._flow._main import __instance__
from quanta.config import settings
config = settings('flow').cap




class Series(pd.Series):
    _internal_names = pd.Series._internal_names + []
    _internal_names_set = set(_internal_names)
    _metadata = pd.Series._metadata  + [f'_{i}' for i in config.recommand_settings.keys()]

    @classmethod
    @lru_cache(maxsize=16)
    def _get_values(cls, portfolio_type, key, name):
        x = __instance__[portfolio_type](key).loc[name]
        return x
    def _lazymem_clean(self):
        attrs = {i for i in self.__dict__.keys() if '__lazymem_' in i}
        [delattr(self, i) for i in attrs]

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_sliced(self):
        return Series    
    
    def __init__(self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        fastpath=False,
        **kwargs
    ):
        params = config.recommand_settings.to_dict() | kwargs
        [setattr(self, f'_{i}',j) for i,j in params.items()]
        super().__init__(data, index, dtype, name, copy, fastpath)
        
    def __add__(self, other: pd.Series) -> 'Series':
        if isinstance(other, pd.Series):
            index = self.index.union(other.index)
            self = self.reindex(index, fill_value=0)
            other = other.reindex(index, fill_value=0)
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit isnot match...')
            x = super().__add__(other)
            x.cash = x.cash + getattr(other, 'cash', 0)
            x.name = max(self.name,pd.to_datetime(other.name))
        else:
            x = super().__add__(other)
        return x    
    
    def __radd__(self, other: pd.Series) -> 'Series':
        return self.__add__(other)
    
    def __sub__(self, other: pd.Series) -> 'Series':
        if isinstance(other, pd.Series):
            index = self.index.union(other.index)
            self = self.reindex(index, fill_value=0)
            other = other.reindex(index, fill_value=0)
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit is not match...')
            x = super().__sub__(other)
            x.cash = x.cash + getattr(other, 'cash', 0)
            x.name = max(self.name,pd.to_datetime(other.name))
        else:
            x = super().__sub__(other)
        return x    
    
    def __rsub__(self, other: pd.Series) -> 'Series':
        return self.__sub__(other)
    
    def __mul__(self, others: [int, float, np.number, pd.Series]) -> 'Series':
        if isinstance(others, (int, float, np.number)):
            x = super().__mul__(others)
            x.cash = x.cash * others
        else:
            x = super().__mul__(others)
        x.name = self.name
        return x
    
    def __truediv__(self, others: [int, float, np.number, pd.Series]) -> 'Series':
        if isinstance(others, (int, float, np.number)):
            x = super().__truediv__(others)
            x.cash = x.cash / others
        else:
            x = super().__truediv__(others)
        x.name = self.name
        return x
    
    @property
    def name(self) -> pd.Timestamp:
        return self._name
    @name.setter
    def name(self, trade_dt: pd.Timestamp):
        trade_dt = pd.to_datetime(trade_dt)
        if (getattr(self, '_name', None) is not None
            and (self._unit == 'share') 
            and (self._is_adj == False)
        ):
            post_adj = (
                self._get_values(
                    self.portfolio_type, config.post_factor, trade_dt
                ) /
                self._get_values(
                    self.portfolio_type, config.post_factor, self.name
                )
            )
            self.values[:] = self.values * post_adj
        self._lazymem_clean()
        self._name = trade_dt
                    
    @property
    def cash(self):
        if self.state != 'trade':
            return self._cash
        else:
            if not hasattr(self, '__lazymem_cash__'):
                x = (self * self._get_values(
                    self.portfolio_type, 
                    config.trade_price + ('_adj' if self.is_adj else ''), 
                    self._name
                )).sum() * -1
                setattr(self, '__lazymem_cash__', x)
            return getattr(self, '__lazymem_cash__')
    @cash.setter
    def cash(self, v):
        self._cash = v
        
    @property
    def is_adj(self):
        return self._is_adj
    @is_adj.setter
    def is_adj(self, v):
        if (self.unit == 'share') and (self._is_adj != v):
            post_adj =  self._get_values(
                self.portfolio_type, config.post_factor, self.name
            )
            self.values[:] = self.values * post_adj
        self._lazymem_clean()
        self._is_adj = v
        
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, v):
        assert v in config.state_types
        self._lazymem_clean()
        if (self._state == 'signal') and (v != 'signal'):
            self.f.day_shift(n=1, copy=False)
            self._cash = 0
        
        self._state = v
        
    def __signal_to_order__(self):
        self.f.day_shift(n=1, copy=False)
        self._cash = 0
    
    def __order_to_trade__(self):
        self._state = 'trade'
        
    def __trade_to_settle__(self):
        self._state = 'settle'
    
        
    @property
    def unit(self):
        return self._unit
    @unit.setter
    def unit(self, v):
        assert v in config.unit_types
        getattr(self, f'__{self._unit}_to_{v}__')(v)
        self._unit = v
        

        
    
        
    
    @property    
    def portfolio_type(self):
        return self.index.name.split('_')[0]
    
    def current(self, is_adj=True):
        key = self._price_mapping.get(self._state)
        is_adj = self._use_adj_price if is_adj is None else is_adj
        key = key +'_adj' if is_adj else key
        x = self._get_values(self.portfolio_type, key, self.name)
        return x

                        
    
    
    
    
    
    
            
