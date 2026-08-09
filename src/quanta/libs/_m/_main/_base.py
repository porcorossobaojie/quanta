# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:51:56 2026

@author: Porco Rosso
"""
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional, Union, List, Dict, Any

from ._connect import main as meta_table
from ....config import settings
"""
from quanta.libs._m._connect import main as meta_table
from quanta.config import settings
"""

table_info = settings('data').public_keys.recommand_settings

# Global dictionary to map portfolio types to their respective tables
TABLE_DIC = {i: {} for i in table_info.portfolio_types}
TABLES = meta_table.table_info()['table_name'].unique()

for table in TABLES:
    x = meta_table(table=table)
    TABLE_DIC[x.portfolio_type].update({table: x})


class main():
    """
    ===========================================================================
    Main entry point for data flow operations, providing a high-level interface
    to access and manipulate financial data across different portfolio types.
    ---------------------------------------------------------------------------
    数据流操作的主入口, 提供跨不同投资组合类型访问和操作金融数据的高级接口.
    ---------------------------------------------------------------------------
    """

    def __init__(self, portfolio_type: str = 'astock'):
        """
        =======================================================================
        Initializes the flow main instance for a specific portfolio type.

        Parameters
        ----------
        portfolio_type : str
            The type of portfolio to operate on (e.g., 'astock', 'afund').
            Default is 'astock'.
        -----------------------------------------------------------------------
        为特定的投资组合类型初始化流主实例.

        参数
        ----
        portfolio_type : str
            要操作的投资组合类型 (例如 'astock', 'afund'). 默认为 'astock'.
        -----------------------------------------------------------------------
        """
        self.portfolio_type = portfolio_type
        [setattr(self, i, j) for i, j in TABLE_DIC.get(portfolio_type).items()]

    @property
    def _help(self) -> pd.DataFrame:
        """
        =======================================================================
        Returns internal table information filtered by the current
        portfolio type.

        Returns
        -------
        pd.DataFrame
            DataFrame containing metadata for relevant tables.
        -----------------------------------------------------------------------
        返回按当前投资组合类型过滤的内部表信息.

        返回
        ----
        pd.DataFrame
            包含相关表元数据的 DataFrame.
        -----------------------------------------------------------------------
        """
        x = meta_table().table_info()
        x = x[x.iloc[:, -3].str.contains(self.portfolio_type)]
        return x

    def help(self, col: str) -> pd.DataFrame:
        """
        =======================================================================
        Finds table and column information for a given column name.

        Parameters
        ----------
        col : str
            The column name to search for.

        Returns
        -------
        pd.DataFrame
            Information about which tables contain the specified column.
        -----------------------------------------------------------------------
        查找给定列名的表和列信息.

        参数
        ----
        col : str
            要搜索的列名.

        返回
        ----
        pd.DataFrame
            关于哪些表包含指定列的信息.
        -----------------------------------------------------------------------
        """
        return meta_table.__find__(col, self._help)

    def __columns_to_tables__(
        self,
        columns: Union[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        =======================================================================
        Maps a list of column names to their respective source tables.

        Parameters
        ----------
        columns : Union[str, List[str]]
            Single column name or list of column names.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where keys are table names and values are lists of
            columns belonging to those tables.
        -----------------------------------------------------------------------
        将列名列表映射到其各自的源表.

        参数
        ----
        columns : Union[str, List[str]]
            单个列名或列名列表.

        返回
        ----
        Dict[str, List[str]]
            一个字典, 键为表名, 值为属于这些表的列列表.
        -----------------------------------------------------------------------
        """
        tables = self._help
        if isinstance(columns, str) and len(columns.split('-')) > 1:
            index = columns.split('-')
            index = tables[tables.iloc[:, -3].str.contains(index[0]) & (tables.iloc[:, -2] == index[1])]
        else:
            columns = [columns] if isinstance(columns, str) else [i for i in columns]
            index = [tables[tables.iloc[:, -2] == i].index for i in columns]
            index = index[0].append(index[1:])
            index = tables.loc[index]
        index = index.groupby(index.columns[-3])[index.columns[-2]].apply(list).to_dict()
        return index

    @property
    def window(self):
        if not hasattr(self, '_window'):
            self._window = 1
        return self._window

    @window.setter
    def window(self, v):
        if self.window != v:
            tables = TABLE_DIC.get(self.portfolio_type).keys()
            for i in tables:
                table_instance = getattr(self, i)
                table_instance.window = v
                setattr(self, i, table_instance)
            self._window = v

    def __call__(
        self,
        end,
        window = 1,
        start = None,
        table = 'eodprices',
        **kwargs: Any
        ) -> Union[pd.Series, pd.DataFrame]:
        self.window = window
        table = getattr(self, f"{self.portfolio_type}{table}")
        return table(start, end, window)

    def shift(self, n=1, calendar='trade_days', table='eodprices'):
        last_day = max(getattr(self, f"{self.portfolio_type}{table}").internal_data.keys())
        if n >= 0:
            day_list = getattr(getattr(self, f"{self.portfolio_type}{table}").calendar, calendar).loc[last_day:]
            if len(day_list) > n:
                end = day_list.iloc[n]
            else:
                raise ValueError(f"max of last day is: {day_list.iloc[-1]}, last day now is: {last_day}, n is: {n}")
        else:
            day_list = getattr(getattr(self, f"{self.portfolio_type}{table}").calendar, calendar).loc[:last_day].iloc[:-1]
            if len(day_list) > -n:
                end = day_list.iloc[n]
            else:
                raise ValueError(f"max of last day is: {day_list.iloc[-1]}, last day now is: {last_day}, n is: {n}")
        return self.__call__(end, window=self.window, table=table)
