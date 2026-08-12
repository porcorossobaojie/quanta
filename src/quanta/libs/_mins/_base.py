# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:07:35 2026

@author: Porco Rosso
"""
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd

from quanta.libs._mins._connect import main as meta_table
from quanta.config import settings

TABLE_DIC = {i: {} for i in settings('data').public_keys.minfreq_settings.portfolio_types}
table_info = settings('libs').db.DuckDB.recommand_settings
TABLES = [i.name for i in (Path(table_info.path) / table_info.parquet).iterdir() if i.is_dir()]

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

    def __init__(self, portfolio_type: str = 'astock') -> None:
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
    def window(self) -> int:
        """Returns the current rolling window size | 返回当前滚动窗口大小"""
        if not hasattr(self, '_window'):
            self._window = 1
        return self._window

    @window.setter
    def window(self, v: int) -> None:
        """Sets the rolling window across all tables | 为所有表设置滚动窗口"""
        if self.window != v:
            tables = TABLE_DIC.get(self.portfolio_type).keys()
            for i in tables:
                table_instance = getattr(self, i)
                table_instance.window = v
                setattr(self, i, table_instance)
            self._window = v

    def __call__(
        self,
        end: Union[str, pd.Timestamp],
        window: int = 1,
        start: Optional[Union[str, pd.Timestamp]] = None,
        table: str = 'eodprices',
        **kwargs: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        =======================================================================
        Reads data ending at the given day with a rolling window.

        Parameters
        ----------
        end : Union[str, pd.Timestamp]
            The end trading day.
        window : int
            The number of days to include. Default is 1.
        start : Optional[Union[str, pd.Timestamp]]
            The optional start bound. Default is None.
        table : str
            The table suffix (e.g., 'eodprices'). Default is 'eodprices'.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            The requested data slice.
        -----------------------------------------------------------------------
        读取以给定日期结束且带滚动窗口的数据.

        参数
        ----
        end : Union[str, pd.Timestamp]
            结束交易日.
        window : int
            包含的天数. 默认为 1.
        start : Optional[Union[str, pd.Timestamp]]
            可选的起始边界. 默认为 None.
        table : str
            表后缀 (例如 'eodprices'). 默认为 'eodprices'.
        **kwargs : Any
            额外的关键字参数.

        返回
        ----
        Union[pd.Series, pd.DataFrame]
            请求的数据切片.
        -----------------------------------------------------------------------
        """
        self.window = window
        table = getattr(self, f"{self.portfolio_type}{table}")
        return table(start, end, window)

    def shift(
        self,
        n: int = 1,
        calendar: str = 'trade_days',
        table: str = 'eodprices'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        =======================================================================
        Shifts the end day by n trading days and reads the corresponding data.

        Parameters
        ----------
        n : int
            The number of days to shift (negative for backward). Default is 1.
        calendar : str
            The calendar day set to use. Default is 'trade_days'.
        table : str
            The table suffix. Default is 'eodprices'.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            The data slice ending at the shifted day.
        -----------------------------------------------------------------------
        将结束日期平移 n 个交易日后读取对应数据.

        参数
        ----
        n : int
            平移的天数 (负数表示向后). 默认为 1.
        calendar : str
            使用的日历日期集. 默认为 'trade_days'.
        table : str
            表后缀. 默认为 'eodprices'.

        返回
        ----
        Union[pd.Series, pd.DataFrame]
            以平移后日期结束的数据切片.
        -----------------------------------------------------------------------
        """
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
