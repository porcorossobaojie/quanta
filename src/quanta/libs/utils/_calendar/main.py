# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:29:07 2026

@author: Porco Rosso
"""
from functools import lru_cache

import pandas as pd
from ....config import settings, login_info
from ...db.main import main as db


class main():
      
    def __init__(
        self, 
        start, 
        bias = '19 hours', 
        daily_bias = '15 hours', 
        name = 'trade_dt', 
        baktable = 'astockeodprices'
    ):
        self.start = pd.to_datetime(start)
        self.bias = bias
        self.daily_bias = daily_bias
        self.name = name
        self.baktable = baktable
        
    @property
    def calendar_days(self):
        @lru_cache(maxsize=1)       
        def _internal(start, bias, daily_bias, name):
            calendar_days = pd.date_range(
                start = start,
                freq = 'd',
                end = pd.Timestamp.today() - pd.Timedelta(bias),
                name = name)
            if daily_bias is not None:
                calendar_days = calendar_days + pd.Timedelta(daily_bias)
            calendar_days = pd.Series(calendar_days, index = calendar_days)
            return calendar_days
        x = _internal(self.start, self.bias, self.daily_bias, self.name)
        return x

    @property
    def trade_days(self):
        @lru_cache(maxsize=1)
        def _internal(start, bias, daily_bias, name):
            try:
                trade_days = db().__read__(
                    table = self.baktable, 
                    columns = name).iloc[:, 0] - pd.Timedelta(daily_bias)
                trade_days = pd.to_datetime(
                    sorted(
                        trade_days[trade_days >= pd.to_datetime(self.start)].unique()
                    )
                )
            except Exception:
                import jqdatasdk as jq
                jq.auth(**login_info('account').joinquant)
                trade_days = pd.to_datetime(
                    jq.get_trade_days(
                        start,
                        pd.Timestamp.today() - pd.Timedelta(bias)
                    )
                )
                trade_days.name = name
            if daily_bias is not None:
                trade_days = trade_days + pd.Timedelta(daily_bias)
            trade_days =  pd.Series(trade_days, index = trade_days)
            return trade_days
        x = _internal(self.start, self.bias, self.daily_bias, self.name)
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
    
    
        
        
