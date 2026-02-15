# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:51:56 2026

@author: Porco Rosso
"""
import pandas as pd
from functools import reduce

from quanta.libs._flow._connect import main as meta_table
from quanta.config import settings
config = settings('data').public_keys.recommand_settings

TABLE_DIC = {i:{} for i in config.portfolio_types}
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
            columns = [columns.lower()] if isinstance(columns, str) else [i.lower() for i in columns]
            index = [tables[tables.iloc[:, -2] == i].index for i in columns]
            index = reduce(lambda a, b: a.union(b), index)
            index = tables.loc[index]
        index = index.groupby(index.columns[-3])[index.columns[-2]].apply(list).to_dict()
        return index

    def __call__(self, columns):
        dic = self.__columns_to_tables__(columns)
        if len(dic) - 1:
            df = {i:getattr(self, i)(j) for i,j in dic.items()}
            df = pd.concat(df, axis=1)
        else:
            df = [getattr(self, i)(j[0] if len(j) == 1 else j) for i,j in dic.items()][0]

        if df.columns.nlevels > 1:
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







