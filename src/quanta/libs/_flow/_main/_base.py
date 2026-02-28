# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:51:56 2026

@author: Porco Rosso
"""
import pandas as pd
from functools import reduce
from functools import lru_cache

from ._connect import main as meta_table, trade_days, calendar_days
from ....config import settings
table_info = settings('data').public_keys.recommand_settings
config = settings('flow')

TABLE_DIC = {i:{} for i in table_info.portfolio_types}
TABLES = meta_table.table_info()['table_name'].unique()
for table in TABLES:
    x = meta_table(table=table)
    TABLE_DIC[x.portfolio_type].update({table: x})



class main():
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

    def __columns_to_tables__(self, columns):
        tables = self._help
        if isinstance(columns, str) and len(columns.split('.')) > 1:
            index = columns.split('.')
            index = tables[tables.iloc[:, -3].str.contains(index[0]) & (tables.iloc[:, -2] == index[1])]
        else:
            columns = [columns] if isinstance(columns, str) else [i for i in columns]
            index = [tables[tables.iloc[:, -2] == i].index for i in columns]
            index = index[0].append(index[1:])
            #index = reduce(lambda a, b: a.append(b), index)
            index = tables.loc[index]
        index = index.groupby(index.columns[-3])[index.columns[-2]].apply(list).to_dict()
        return index

    def __call__(self, *columns):
        dic = self.__columns_to_tables__(columns)
        if len(dic) - 1:
            df = {i:getattr(self, i)(j) for i,j in dic.items()}
            df = pd.concat(df, axis=1)
        else:
            df = [getattr(self, i)(j[0] if len(j) == 1 else j) for i,j in dic.items()][0]

        if isinstance(df, pd.Series) or (isinstance(df, pd.DataFrame) & (df.columns.nlevels > 1)):
            return df
        else:
            for i in range(0, df.columns.nlevels, -1):
                if df.columns.get_level_values(i).unique() == 1:
                    df.columns = df.colums.droplevel(i)
            return df

    def finance(
        self,
        column,
        quarter_adj: bool = False,
        quarter_diff: int = 1,
        shift: int = 0,
        periods: int = 1,
        min_periods = None,
        drop_zero = True,
    ):
        dic = self.__columns_to_tables__(column)
        df = getattr(self, list(dic.keys())[0]).__finance__(list(dic.values())[0][0], quarter_adj, quarter_diff, shift, periods, min_periods, drop_zero)
        return df

    @lru_cache(maxsize=8)
    def multilize(self, column, value_column=None):
        if value_column is None:
            df = self(column)
            if not df.index.duplicated().any():
                df = df.stack()
            df = df.to_frame(column)
            df['temp_value'] = 1
        else:
            df = self(column, value_column)
        df = df.set_index(column, append=True)
        df = df.iloc[:, 0].unstack(0).T
        if df.columns.names[0] in [table_info.key.astock_code, table_info.key.afund_code]:
            df.columns = df.columns.swaplevel(-1,0)
        if value_column is None:
            df = df.notnull()
        df = df.sort_index(axis=1).sort_index()
        return df

    @lru_cache(maxsize=8)
    def _astock_listing(self, limit=126):
        if not hasattr(self, '_internal_listing'):
            df = (self(config.listing.astock_listing_date, config.listing.astock_delisting_date).clip(
                upper = pd.to_datetime(pd.Timestamp.today().date()))
                .set_index(config.listing.astock_delisting_date, append=True)[config.listing.astock_listing_date]).unstack(0)
            df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='d')).bfill().dropna(how='all', axis=1)
            df.index = df.index + pd.Timedelta(meta_table.time_bias)
            df = df.reindex(trade_days)
            x = df[df.sub(df.index, axis=0).astype('int64') <= 0].notnull().cumsum()
            setattr(self, '_internal_listing', x)
        x =  getattr(self, '_internal_listing')
        x = x >= limit
        return x

    @lru_cache(maxsize=8)
    def _afund_listing(self, limit=126):
        if not hasattr(self, '_internal_listing'):
            x = self(config.listing.afund_listing_date).notnull().cumsum()
            setattr(self, '_internal_listing', x)
        x =  getattr(self, '_internal_listing')
        x = x >= limit
        return x

    def listing(self, limit=126):
        func = getattr(self, f"_{self.portfolio_type}_listing")
        df = func(limit)
        return df

    @lru_cache(maxsize=1)
    def not_st(self, value=1):
        key = config.status.not_st
        dic = self.__columns_to_tables__(key)
        table_obj = [getattr(self, i) for i,j in dic.items()][0]
        df = table_obj.__read__().set_index(table_obj.index_keys)[key]
        status = {301001:0, 301002:1, 301003:2, 301005:3, 301006:4}
        df = df.replace(status)
        df = df[df.isin(status.values())]
        df = df.loc[~df.index.duplicated(keep = 'last')].unstack(table_obj.code).sort_index(axis=1).sort_index()
        df = df.ffill().reindex(calendar_days).ffill().reindex(trade_days).loc[meta_table.start_date:]
        df = df < value
        return df

    @lru_cache(maxsize=1)
    def traced_index(self, column='traced_index_name'):
        df = self('end_date', column)
        df['end_date'] = (pd.to_datetime(df['end_date']) + meta_table.time_bias).fillna(trade_days[-1]).dropna()
        x = df.reset_index().drop_duplicates(keep='first', subset=df.index.name).dropna()
        x = x.pivot(index='end_date', columns=df.index.name, values=column).reindex(trade_days)
        x = x.loc[meta_table.start_date:].bfill()
        return x
