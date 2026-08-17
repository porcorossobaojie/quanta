# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:42:44 2026

@author: Porco Rosso
"""
from pathlib import Path
from functools import lru_cache
from cachetools import LRUCache
from typing import Literal, Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from quanta.libs.db.main import main as db
from quanta.config import settings
from quanta.libs.utils import calendar as cal
"""

from ...db.main import main as db
from ....config import settings
from ....libs.utils import calendar as cal

"""
config = settings('data').public_keys
columns_info = config.minfreq_settings.key

class main(type('recommand_settings', (), config.minfreq_settings.key), db):
    """
    ===========================================================================
    A specialized database connection class for financial data flow,
    inheriting from the base database class and dynamic public keys.
    ---------------------------------------------------------------------------
    用于金融数据流的专用数据库连接类, 继承自基础数据库类和动态公共键.
    ---------------------------------------------------------------------------
    """
    date_start = pd.to_datetime(config.minfreq_settings.key.date_start)
    calendar = cal(
        start = config.minfreq_settings.key.date_start,
        daily_bias = None,
        name = columns_info.trade_dt,
        baktable = 'aindexeodprices'
    )

    @property
    def portfolio_type(self) -> Optional[str]:
        """
        =======================================================================
        Determines the portfolio type based on the table name.

        Returns
        -------
        Optional[str]
            The identified portfolio type (e.g., 'astock', 'afund').
        -----------------------------------------------------------------------
        根据表名确定投资组合类型.

        返回
        ----
        Optional[str]
            识别出的投资组合类型 (例如 'astock', 'afund').
        -----------------------------------------------------------------------
        """
        for i in config.recommand_settings.portfolio_types:
            if i in self.table:
                return i
        return 'other'

    @property
    def code(self) -> str:
        """
        =======================================================================
        Returns the specific code column name for the current portfolio type.

        Returns
        -------
        str
            The code column name (e.g., 'astock_code').
        -----------------------------------------------------------------------
        返回当前投资组合类型的特定代码列名.

        返回
        ----
        str
            代码列名 (例如 'astock_code').
        -----------------------------------------------------------------------
        """
        attr = f"{self.portfolio_type}_code"
        return attr

    def __read_from_db__(
        self, 
        date,
        ):
        """
        =======================================================================
        Reads the daily data for a given date from the parquet store and pivots
        it into a wide DataFrame indexed by the trading datetime.

        Parameters
        ----------
        date : str
            The trading date partition to read.

        Returns
        -------
        pd.DataFrame
            The pivoted DataFrame with datetime columns.
        -----------------------------------------------------------------------
        从 parquet 存储中读取指定日期的数据, 并将其透视为以交易时间为列的宽表.

        参数
        ----
        date : str
            需要读取的交易日期分区.

        返回
        ----
        pd.DataFrame
            透视后的 DataFrame, 列为交易时间.
        -----------------------------------------------------------------------
        """
        path = Path(self.path) / self.parquet / self.table /f"date={date}"
        df = self.__read_parquet__(path)
        df = df.T
        return df
    
    @property
    def window(self) -> int:
        """Returns the current rolling window size | 返回当前滚动窗口大小"""
        if not hasattr(self, '_internal_data'):
            self._internal_data = LRUCache(1)   
            return 1
        else:
            return self._internal_data.maxsize

    @window.setter
    def window(self, v: int) -> None:
        """Sets the window size and resizes the internal cache | 设置窗口大小并调整内部缓存"""
        if  (v is not None) & (v != self.window):
            x = self.internal_data
            self._internal_data = LRUCache(v)
            self._internal_data.update(x)

    @property
    def start(self):
        return min(self._internal_data.keys()) if len(self._internal_data.keys()) else None
    
    @property
    def end(self):
        return max(self._internal_data.keys()) if len(self._internal_data.keys()) else None

    @property
    def internal_data(self) -> LRUCache:
        """Returns the internal LRU data cache | 返回内部 LRU 数据缓存"""
        if not hasattr(self, '_internal_data'):
            self._internal_data = LRUCache(1)
        return self._internal_data
        
    def __read_from_internal__(
        self,
        dates: pd.Series
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Reads data for multiple dates, falling back to the database | 读取多日数据, 缺失时回退到数据库"""
        dic = {}
        for i in dates:
            j = self.internal_data.get(i)
            if j is None:
                j = self.__read_from_db__(pd.to_datetime(i).date())
            dic[i] = j
        if not hasattr(self, '_internal_data'):
            self._internal_data = LRUCache(max([len(dic.keys()), 1]))
        else:
            if self._internal_data.maxsize < len(dic):
                self._internal_data = LRUCache(len(dic))
        self._internal_data.update(dic)
        return dic

    def __call__(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        window: int = 1
    ) -> pd.DataFrame:
        """
        =======================================================================
        Reads and concatenates data for the requested day range.

        Parameters
        ----------
        start : Optional[Union[str, pd.Timestamp]]
            The start bound. Default is None.
        end : Optional[Union[str, pd.Timestamp]]
            The end bound. Default is None.
        window : int
            The number of days to read. Default is 1.

        Returns
        -------
        pd.DataFrame
            The concatenated data.
        -----------------------------------------------------------------------
        读取并拼接所请求日期范围内的数据.

        参数
        ----
        start : Optional[Union[str, pd.Timestamp]]
            起始边界. 默认为 None.
        end : Optional[Union[str, pd.Timestamp]]
            结束边界. 默认为 None.
        window : int
            读取的天数. 默认为 1.

        返回
        ----
        pd.DataFrame
            拼接后的数据.
        -----------------------------------------------------------------------
        """
        dates = self.calendar.units(start, end, window)
        x = self.__read_from_internal__(dates)
        x = list(x.values())
        x = x[0] if len(x) == 1 else pd.concat(x)
        self.window = window
        return x        
        
        
        
