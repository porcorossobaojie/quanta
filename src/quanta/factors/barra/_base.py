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
        bench = 'full',
        portfolio_type = 'astock'
    ):
        bench = cls.bench(bench).to_frame().astype('float32')
        bench_values = bench.values
        ret = getattr(flow, portfolio_type)(cls.returns).f.filtered(listing_limit=1).astype('float32')
        ret_values = ret.values
        w = pd.tools.half_life(periods, periods//4)[np.newaxis, :].astype('float32')
        w_matrix = w.repeat(ret.shape[1]).reshape(-1, ret.shape[1])
        filter_bool = ret.notnull().astype('float32')
        filter_bool_values = filter_bool.values
        ret_filled = ret.fillna(0).astype('float32')
        ret_filled_values = ret_filled.values
        filter_bool_count = filter_bool.rolling(periods, min_periods=periods//4).sum().values.astype('float32')
        
        alpha = np.zeros((ret.index[periods:].shape[0], ret.shape[1]))
        beta = np.zeros_like(alpha)
        resid = np.zeros_like(alpha)
        
        for i in range(ret.shape[0] - periods):
            temp_ret_values = ret_values[i:i+periods]
            ret_w_sum = w @ ret_filled_values[i: i+periods]
            w_sum = np.sum(np.where(filter_bool_values[i: i+periods], w_matrix, 0), axis=0)
            w_sum = np.where(w_sum != 0 , w_sum, np.nan)
            ret_w_mean = ret_w_sum / w_sum[np.newaxis, :]
            bench_var = bench_values[i: i+periods] - (w @(bench_values[i: i+periods])) / w_sum[np.newaxis, :]
            ret_var = ret.values[i: i+periods] - ret_w_mean
            ret_var_filled = np.nan_to_num(ret_var * bench_var, 0)
            
            beta_i = (w @ ret_var_filled) / (w @ np.nan_to_num(ret_var ** 2, 0))
            bench_prem = beta_i * bench_values[i:i+periods]
            alpha_i = np.sum(np.nan_to_num(temp_ret_values - bench_prem, 0), axis=0) / filter_bool_count[i + periods - 1]
            std_i = np.sum(np.nan_to_num((temp_ret_values - alpha_i - bench_prem) ** 2, 0) * w, axis=0) / w_sum
            
            alpha[i] = alpha_i[0]
            beta[i] = beta_i[0]
            resid[i] = std_i[0]
        
        alpha = pd.DataFrame(alpha, index=ret.index[periods:], columns = ret.columns).f.filtered(listing_limit=periods)
        beta = pd.DataFrame(beta, index=ret.index[periods:], columns = ret.columns).f.filtered(listing_limit=periods)
        resid = pd.DataFrame(resid, index=ret.index[periods:], columns = ret.columns).f.filtered(listing_limit=periods)
        return alpha, beta, resid
    
    @classmethod
    def beta(cls, periods=252, bench='full', portfolio_type='astock'):
        return cls._beta(periods, bench)[1]
    
            
        

        
        
        
