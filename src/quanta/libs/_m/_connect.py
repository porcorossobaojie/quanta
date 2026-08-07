# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:11:56 2026

@author: Porco Rosso
"""

from functools import lru_cache
from cachetools import LRUCache
from typing import Literal, Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from quanta.libs.db.main import main as db
from quanta.config import settings, login_info
from quanta.libs.utils import calendar as cal
"""
from ...utils import calendar as cal
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
        start = pd.to_datetime(config.minfreq_settings.key.date_start), 
        daily_bias = None,
        name = columns_info.trade_dt,
        baktable = 'aindexeodprices'
    )

    @classmethod
    @lru_cache(maxsize=1)
    def table_info(cls) -> pd.DataFrame:
        """
        =======================================================================
        Retrieves cached schema information for all tables.

        Returns
        -------
        pd.DataFrame
            DataFrame containing schema metadata.
        -----------------------------------------------------------------------
        获取所有表的缓存模式信息.

        返回
        ----
        pd.DataFrame
            包含模式元数据的 DataFrame.
        -----------------------------------------------------------------------
        """
        return cls.__schema_info__()
    
    @classmethod    
    def __get_date_from_parameters__(cls, start=None, end=None, window=None):
        pass
        

    @property
    def columns(self) -> List[str]:
        """
        =======================================================================
        Returns the list of column names for the current table.

        Returns
        -------
        List[str]
            List of column names.
        -----------------------------------------------------------------------
        返回当前表的列名列表.

        返回
        ----
        List[str]
            列名列表.
        -----------------------------------------------------------------------
        """
        x = self.table_info()
        x = x[x['table_name'] == self.table]
        return x['column_name'].to_list()

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
        return None

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
        return getattr(self, attr)

    @property
    def index_keys(self) -> List[str]:
        """
        =======================================================================
        Identifies the primary index keys present in the current table.

        Returns
        -------
        List[str]
            List of index column names.
        -----------------------------------------------------------------------
        识别当前表中存在的主要索引键.

        返回
        ----
        List[str]
            索引列名列表.
        -----------------------------------------------------------------------
        """
        keys = [self.trade_dt, self.code]
        return keys

    @property
    def filter_key(self) -> str:
        """
        =======================================================================
        Returns the primary time-based filtering key for the current table.

        Returns
        -------
        str
            The filtering column name.
        -----------------------------------------------------------------------
        返回当前表的主要基于时间的过滤键.

        返回
        ----
        str
            过滤列名.
        -----------------------------------------------------------------------
        """
        return self.index_keys[0]

    def __columns_standard__(self, columns: Union[str, List[Any]]) -> List[str]:
        """
        =======================================================================
        Validates and standardizes a list of column names.

        Parameters
        ----------
        columns : Union[str, List[Any]]
            Single column name or list of columns to validate.

        Returns
        -------
        List[str]
            Standardized list of valid column names.

        Raises
        ------
        ValueError
            If any column name is invalid for the current table.
        -----------------------------------------------------------------------
        验证并标准化列名列表.

        参数
        ----
        columns : Union[str, List[Any]]
            要验证的单个列名或列列表.

        返回
        ----
        List[str]
            标准化后的有效列名列表.

        异常
        ----
        ValueError
            如果任何列名对于当前表无效.
        -----------------------------------------------------------------------
        """
        columns = [columns] if isinstance(columns, str) else [str(i) for i in columns]
        not_have_columns = [i for i in columns if i not in self.columns]
        if not len(not_have_columns):
            return columns
        else:
            raise ValueError(f"Invalid value '{not_have_columns}' for parameter 'columns'. Valid values are: {self.columns}")

    def __read_from_db__(
        self,
        date, 
        **kwargs: Any
    ) -> Optional[Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        =======================================================================
        Loads data from the database into an internal cache and reshapes it
        if necessary.

        Parameters
        ----------
        returns : bool
            Whether to return the loaded data immediately. Default is False.
        **kwargs : Any
            Additional arguments for the database read method.

        Returns
        -------
        Optional[Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]]
            The cached data if returns is True.
        -----------------------------------------------------------------------
        从数据库加载数据到内部缓存, 并在必要时对其进行重塑.

        参数
        ----
        returns : bool
            是否立即返回加载的数据. 默认为 False.
        **kwargs : Any
            数据库读取方法的附加参数.

        返回
        ----
        Optional[Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]]
            如果 returns 为 True, 则返回缓存的数据.
        -----------------------------------------------------------------------
        """
        df = self.__read__(where=f"{self.trade_dt} >= {date} and {self.trade_dt} < {pd.to_datetime(date) + pd.Timedelta(1, 'd')}", show_time=True)
        df.columns = pd.CategoricalIndex(df.columns)
        df[self.code] = pd.CategoricalIndex(df[self.code])
        df = df.set_index(self.index_keys).unstack()
        df.columns
        return df
    
    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, v):
        self._window = v
        self._internal_data = self._internal_data(v)
        
    @property
    def internal_data(self):
        if not hasattr(self, '_internal_data'):
            self._internal_data = LRUCache(1)
        return self._internal_data
    

    
