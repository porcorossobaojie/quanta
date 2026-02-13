# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:11:56 2026

@author: Porco Rosso
"""

from functools import lru_cache
from typing import Literal
import pandas as pd

from quanta.libs.db.main import main as db
from quanta.config import settings
config = settings('data').public_keys

class main(db, type('public_keys', (), config.recommand_settings.key)):
    time_bias =  pd.Timedelta(config.recommand_settings.time_bias)
    start_date = pd.to_datetime(settings('flow').start_date) + pd.Timedelta(config.recommand_settings.time_bias)
    
    @classmethod
    @lru_cache(maxsize=1)
    def table_info(cls):
        return cls.__schema_info__()
    
    @property
    def columns(self):
        x = self.table_info()
        x = x[x['table_name'] == self.table]
        return x['column_name'].to_list()
    
    def __get_from_db__(self, table=None):
        if not hasattr(self, '_internal_data'):
            df = self.__read__(table=self.table if table is None else table)
        
        
table = None
self = main(table='afundeodprices')
df = self.__read__(table=self.table if table is None else table)
