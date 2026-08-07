# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:29:07 2026

@author: Porco Rosso
"""
from functools import lru_cache

import pandas as pd
from quanta.config import settings, login_info
from quanta.libs.db.main import main as db

config = settings('data').public_keys.recommand_settings.key

class main():
      
    def __init__(self, start, bias='19 hours', daily_bias=True, trade_days_baktable = 'astockeodprices'):
        self.start = pd.to_datetime(start)
        self.bias = bias
        self.daily_bias = daily_bias
        self.baktable = trade_days_baktable
        
    @property
    def calendar_days(self):
        @lru_cache(maxsize=1)       
        def _internal(start, bias, daily_bias):
            calendar_days = pd.date_range(
                start = start,
                freq = 'd',
                end = pd.Timestamp.today() - pd.Timedelta(bias),
                name = config.trade_dt)
            if daily_bias:
                calendar_days = calendar_days + pd.Timedelta(config.time_bias)
            calendar_days = pd.Series(calendar_days, index = calendar_days)
        x = _internal(self.start, self.bias, self.daily_bias)
        return x

    @property
    def trade_days(self):
        @lru_cache(maxsize=1)
        def _internal(start, bias, daily_bias):
            try:
                import jqdatasdk as jq
                jq.auth(**login_info('account').joinquant)
                trade_days = pd.to_datetime(
                    jq.get_trade_days(
                        start,
                        pd.Timestamp.today() - pd.Timedelta(bias)
                    )
                )
                trade_days.name = config.trade_dt
            except Exception:
                trade_days = db().__read__(
                    table = self.baktable, 
                    columns = config.trade_dt).iloc[:, 0] - pd.Timedelta((config.time_bias))
                trade_days = pd.to_datetime(
                    sorted(
                        trade_days[trade_days >= pd.to_datetime(self.start)].unique()
                    )
                )
            if self.daily_bias:
                trade_days = trade_days + pd.Timedelta(config.time_bias)
            trade_days =  pd.Series(trade_days, index = trade_days)
            return trade_days
        x = _internal(self.start, self.bias, self.daily_bias)
        return x            

    @property
    def units(self, start=None, end=None, window=None, day_set='trade_days'):
        if start == end == window:
            raise ValueError("<start>, <end>, <window> must have 2 not None values and 1 None value")
        days = getattr(self, day_set)
        if start is not None:
            days = days.loc[pd.to_datetime(start):]
            if window is not None:
                days = days.iloc[:window]
        if end is not None:
            days = days.loc[:pd.to_datetime(end)]
            if window is not None:
                days = days.iloc[-window:]
        return days
    
    
        
        
