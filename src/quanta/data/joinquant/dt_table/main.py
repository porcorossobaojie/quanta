# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:33:11 2026

@author: Porco Rosso
"""

import jqdatasdk as jq
from typing import Any, Literal
import pandas as pd

from quanta.data.joinquant.meta.main import main as meta
from quanta.config import settings
config = settings('data')

class main(meta):
    """
    ===========================================================================

    Main class for handling trade date table data from JoinQuant.

    This class extends the meta class to provide specific data processing
    and daily update functionalities for trade date related tables.

    ---------------------------------------------------------------------------

    处理 JoinQuant 交易日期表数据的主类。

    此类扩展了元类，为交易日期相关表提供特定的数据处理和每日更新功能。

    ---------------------------------------------------------------------------
    """

    def __data_standard_aindexweights__(self, df, **kwargs):
        df = self.__data_standard__(df, **kwargs)
        df[self.trade_dt] = pd.to_datetime(kwargs.get('start_date')) + pd.Timedelta(pd.Timedelta(config.tables.recommand_settings.time_bias))
        df[list(self.columns_information.get('weight').keys())[0]] = df[list(self.columns_information.get('weight').keys())[0]] / 100
        return df

    def __data_standard_astockindustrys__(self, df, **kwargs):
        df.columns = ['_'.join([''.join(i[0].split('_')), i[1].split('_')[-1]]) for i in df.columns]
        df = self.__data_standard__(df, **kwargs)
        return df

    def __data_standard_astockconcept__(self, df, **kwargs):
        df['level_1'] = kwargs.get('start_date')
        df = self.__data_standard__(df, **kwargs)
        return df

    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        df = super().pipeline(**kwargs)

        # construct returns of portfolio by calculated with close price(first use adj price if have)
        if isinstance(self.columns_information, dict):
            ret_key = self.columns_information.get('returns', None)
        else:
            x = eval(f"jq.get_table_info({self.columns_information})")
            x.iloc[:, 0] = x.iloc[:, 0].replace(config.tables.transform | {'code': self.code})
            x.iloc[:, 2] = x.iloc[:, 2].replace({'date': 'datetime', 'DATE': 'datetime'})
            x = x.set_index(x.columns[0]).iloc[:, [1, 0]].T.to_dict('list')
            ret_key = x.get('returns', None)
        if ret_key is not None:
            ret_key = list(ret_key.keys())[0]
            try:
                df[ret_key] = df['close_adj'] / df['preclose_adj'] - 1
            except Exception:
                df[ret_key] = df['close'] / df['preclose'] - 1
        df = df[df.drop([self.trade_dt, self.code], axis=1, errors='ignore').notnull().any(axis=1)]
        return df

    def daily(self, if_exists: Literal['append', 'replace'] = 'append') -> None:
        if self.table == 'astocklisting':  # this table inform the on list time for each stock, which will replace every day
            self.drop_table()
            self.create_table()
            df = self.pipeline()
            self.__write__(df, log=True)

        else:
            if if_exists == 'replace':
                self.drop_table()

            if not self.table_exist():
                getattr(self,  f"create_table_{self.table}", self.create_table)()

            id_key = self.__find_max_of_exist_table__(self.trade_dt)
            days = self._trade_days[self._trade_days > id_key]

            if len(days):
                i = 0
                periods = self.periods
                total = len(days)

                while i < total or jq.get_query_count()['spare'] < 1000000:
                    start_date = days[i]
                    end_idx = min(i + periods - 1, total - 1)
                    end_date = days[end_idx]
                    df = self.pipeline(start_date=str(start_date.date()), end_date=str(end_date.date()))
                    print(f"Update to date: {end_date}; Query count left: {jq.get_query_count()['spare']}")
                    self.__write__(df, log=True)
                    if end_date >= days.max():
                        break
                    i = end_idx + 1

'''
self = main(**config.tables.dt_table.astockconcept)
self.daily()
 '''
