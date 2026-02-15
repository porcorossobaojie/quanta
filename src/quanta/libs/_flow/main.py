# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:51:56 2026

@author: Porco Rosso
"""
import pandas as pd

from quanta.libs._flow._connect import main as meta_table
from quanta.config import settings, login_info
config = settings('data').public_keys.recommand_settings

TABLE_DIC = {i:{} for i in config.portfolio_types}
TABLES = meta_table.table_info()['table_name'].unique()
for table in TABLES:
    x = meta_table(table=table)
    TABLE_DIC[x.portfolio_type].update({table: x})
    
calendar_days = pd.date_range(
    start=pd.to_datetime(meta_table.date_start) + meta_table.time_bias,
    freq='d', 
    end=pd.Timestamp.today() - pd.Timedelta(4, 'h'))

try:
    import jqdatasdk as jq
    jq.auth(**login_info('account').joinquant)
    trade_days = pd.to_datetime(
        jq.get_trade_days(
            meta_table.date_start, 
            pd.Timestamp.today() + pd.offsets.YearEnd(0) - pd.Timedelta(4, 'h') - meta_table.time_bias
            )
        ) + meta_table.time_bias
except:
    trade_days = pd.to_datetime(
        sorted(
            TABLE_DIC.get('astock').get('astockeodprices').__read__(columns=meta_table.trade_dt).iloc[:, 0].unique()
            )
        )
    
class astock():
    def __init__(self, portfolio_type = 'astock'):
        self.portfolio_type = portfolio_type
        [setattr(self, i, j) for i,j in TABLE_DIC.get(portfolio_type).items()]
    
    @property
    def _help(self):
        x = meta_table().table_info()
        x = x[x.iloc[:, -3].str.contains(self.portfolio_type)]
        return x
    
    def help(self, col):
        return meta_table.__find__(col, self._help)
        
