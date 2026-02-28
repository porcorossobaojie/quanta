# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:50:12 2026

@author: Porco Rosso
"""

from functools import lru_cache
import pandas as pd
from ....config import settings
config = settings('flow')

from .core import *

class class_obj:
    merge = merge
setattr(pd, config.extra_pandas_attrname, class_obj)

@pd.api.extensions.register_dataframe_accessor(config.extra_pandas_attrname)
class flow_extra():
    def __init__(self, df_obj):
        self._obj = df_obj
        
    def listing(self, limit, portfolio_type=None):
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = listing(limig, portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]
    
    def not_st(self, value=1, portfolio_type=None):
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = not_st(value, portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]

    def tradestatus(self, portfolio_type=None):
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = statusable(portfolio_type).reindex_like(self._obj).fillna(False)   
        return self._obj[x]

    def filtered(self, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(self._obj).fillna(False)   
        return self._obj[x]
    
    def index_members(self, index_code, invert=False):
        x = index_members(index_code).reindex_like(self._obj).fillna(False)   
        x = ~x if invert else x
        return self._obj[x]
    
    def label(self, code=None, label_df=None, portfolio_type=None):
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = label(code, label_df, portfolio_type)
        reindex_key = list(set(self.obj.columns.names) & set(x.columns.names))[0]
        df = self._obj.reindex(x.columns.get_level_values(reindex_key), axis=1)
        df.collumns = x.columns
        x = x.reindex_like(df).fillna(False)
        return df[x]
    
    def expand(self, portfolio_type='asotck'):
        if portfolio_type == 'astock':
            target_df = label(self._obj.columns.name, portfolio_type)
            x = expand(self._obj, target_df)
            return x
        else:
            raise ValueError('Warning: only support stock -- industry translate...')
            
    def info(self, column, portfolio_type=None):
        return info(self._obj, column, portfolio_type)
    
    @lru_cache(maxsize=64)
    def ic(self, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
        return ic(self._obj, listing_limit, drop_st, tradestatus, portfolio_type)
    
    @lru_cache(maxsize=64)
    def ir(self, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
        x = self.ic(listing_limit, drop_st, tradestatus, portfolio_type)
        return ir(x)
    
    @lru_cache(maxsize=8)
    def port(self, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
        x = port(self._obj, listing_limit, drop_st, tradestatus, portfolio_type)
    
    
    @lru_cache(maxsize=2)
    def test(
        self,
        shift = 0,     
        high = None,
        low = None,
        avgprice = None,
        trade_price = None,
        settle_price = None,
        limit = 0.01,
        trade_cost = 0.0015,
        portfolio_type = None
    ):
        df = self._obj if shift == 0 else self._obj.shift(shift).dropna(how='all')
        obj = qtest(df, high, low, avgprice, trade_price, settle_price, limit, trade_cost, portfolio_type)
        return obj

@pd.api.extensions.register_series_accessor(config.extra_pandas_attrname)
class flow_extra():
    def __init__(self, df_obj):
        self._obj = df_obj       
        
    def info(self, column, portfolio_type=None):
        return series_info(self._obj, column, portfolio_type)
    
    def day_shift(self, shift=1):
        return day_shift(self._obj, shift, True)
    
        
