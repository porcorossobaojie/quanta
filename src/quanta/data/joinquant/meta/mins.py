# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:30:31 2026

@author: Porco Rosso
"""
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import jqdatasdk as jq

from ....libs.utils import merge_dicts, calendar
from ....libs.db.main import main as db
from ....config import settings, login_info

"""
from quanta.libs.utils import merge_dicts, calendar
from quanta.libs.db.main import main as db
from quanta.config import settings, login_info
"""

config = settings('data')

class main(type('recommand_settings', (), config.public_keys.minfreq_settings.key), db):
    """
    ===========================================================================
    Minute-frequency metadata and connection class for JoinQuant data
    extraction, extending the base with a trading calendar and security
    universe.
    ---------------------------------------------------------------------------
    用于 JoinQuant 数据提取的分钟频元数据和连接类, 在基类基础上增加了交易日历
    和证券池.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Initializes the minute-frequency meta instance, setting up the
        environment, trading calendar, and security universe.

        Parameters
        ----------
        **kwargs : Any
            Initial configuration and table parameters.
        -----------------------------------------------------------------------
        初始化分钟频元主实例, 设置环境, 交易日历和证券池.

        参数
        ----
        **kwargs : Any
            初始配置和表参数.
        -----------------------------------------------------------------------
        """
        super().__init__(**kwargs)
        self.__env_init__()
        self.calendar = calendar(self.date_start, daily_bias=self.time_bias, name=self.trade_dt)
        self._stock = jq.get_all_securities('stock', date=None).index.tolist()
        _fund = jq.get_all_securities('fund', date=None)
        self._fund = _fund[_fund.iloc[:, -1] == 'etf'].index.tolist()
        self._index = jq.get_all_securities('index', date=None).index.tolist()
        
    @property
    def trade_days(self):
        """Retrieves the trading days from the internal calendar | 获取内部日历的交易日期"""
        return self.calendar.trade_days
    
    @property
    def portfolio_type(self) -> str:
        """
        =======================================================================
        Identifies the portfolio type from the current table name.

        Returns
        -------
        str
            The portfolio type (e.g., 'astock', 'afund').
        -----------------------------------------------------------------------
        从当前表名中识别投资组合类型.

        返回
        ----
        str
            投资组合类型 (例如 'astock', 'afund').
        -----------------------------------------------------------------------
        """
        for i in config.public_keys.recommand_settings.portfolio_types:
            if i in self.table:
                return i
        return 'other'    
        
    @property
    def code(self) -> str:
        """Retrieves the asset code column name for the current portfolio type | 获取当前投资组合类型的资产代码列名"""
        attr = f"{self.portfolio_type}_code"
        return getattr(self, attr)    

    @property
    def columns(self) -> Dict[str, List[str]]:
        """
        =======================================================================
        Dynamically retrieves and formats column information for the
        current table.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where keys are column names and values are their types.
        -----------------------------------------------------------------------
        动态检索并格式化当前表的列信息.

        返回
        ----
        Dict[str, List[str]]
            一个字典, 其中键是列名, 值是其类型.
        -----------------------------------------------------------------------
        """
        if isinstance(self.columns_information, dict):
            x = merge_dicts(*list(self.columns_information.values()))
        else:
            x = eval(f"jq.get_table_info({self.columns_information})")
            x.iloc[:, 0] = x.iloc[:, 0].replace(config.tables.transform | {'code': self.code})
            x.iloc[:, 2] = x.iloc[:, 2].replace({'date': 'datetime', 'DATE': 'datetime'})
            x = x.set_index(x.columns[0]).iloc[:, [1, 0]].T.to_dict('list')
        return x

    def __columns_rename__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        =======================================================================
        Renames DataFrame columns based on predefined mappings and the current
        table's metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be renamed.

        Returns
        -------
        pd.DataFrame
            The renamed DataFrame.
        -----------------------------------------------------------------------
        根据预定义映射和当前表元数据重命名 DataFrame 列.

        参数
        ----
        df : pd.DataFrame
            要重命名的 DataFrame.

        返回
        ----
        pd.DataFrame
            重命名后的 DataFrame.
        -----------------------------------------------------------------------
        """
        if isinstance(self.columns_information, dict):
            rename_dic = {i: list(j.keys())[0] for i, j in self.columns_information.items()}
        else:
            rename_dic = config.tables.transform | {'code': self.code}
        df = df.reset_index().rename(rename_dic, axis=1)
        df = df.loc[:, df.columns.isin(list(self.columns.keys()))]
        return df
    
    def __get_data_from_jq_remote__(self, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Fetches raw data from JoinQuant remote server using configured
        commands.

        Parameters
        ----------
        **kwargs : Any
            Arguments required by the data fetching command.

        Returns
        -------
        pd.DataFrame
            The raw data from the server.
        -----------------------------------------------------------------------
        使用配置的命令从 JoinQuant 远程服务器获取原始数据.

        参数
        ----
        **kwargs : Any
            数据获取命令所需的参数.

        返回
        ----
        pd.DataFrame
            来自服务器的原始数据.
        -----------------------------------------------------------------------
        """
        df = eval(self.commands.format(**kwargs))
        return df
    
    def __data_standard__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Standardizes raw data, including column renaming, time bias
        adjustments, and data cleaning.

        Parameters
        ----------
        df : pd.DataFrame
            The input raw data.
        **kwargs : Any
            Additional arguments like 'start_date'.

        Returns
        -------
        pd.DataFrame
            Standardized data.
        -----------------------------------------------------------------------
        标准化原始数据, 包括列重命名, 时间偏移调整和数据清洗.

        参数
        ----
        df : pd.DataFrame
            输入的原始数据.
        **kwargs : Any
            包括 'start_date' 在内的附加参数.

        返回
        ----
        pd.DataFrame
            标准化后的数据.
        -----------------------------------------------------------------------
        """
        # df columns renamed as create_table's columns
        df = self.__columns_rename__(df)
        df[self.trade_dt] = pd.to_datetime(df[self.trade_dt])

        # replace nan and standard codes
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        if self.portfolio_type == 'astock':
            df = df[df[self.code].str.contains(r'^\d', na=False)]
        elif self.portfolio_type == 'afund':
            df = df[df[self.code].isin(self._fund)]
        # standard code code means: 000001.xxxx, check need normalize or not
        if (df[self.code].apply(lambda x: len(x)) != 6 + 1 + 4).any():
            df[self.code] = jq.normalize_code(df[self.code].to_list())
        return df
    
    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Executes the full data extraction and standardization pipeline.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for fetching and standardizing.

        Returns
        -------
        pd.DataFrame
            Fully processed data.
        -----------------------------------------------------------------------
        执行完整的数据提取和标准化流水线.

        参数
        ----
        **kwargs : Any
            用于获取和标准化的关键字参数.

        返回
        ----
        pd.DataFrame
            完全处理后的数据.
        -----------------------------------------------------------------------
        """
        df = self.__get_data_from_jq_remote__(**kwargs)
        func = getattr(self, f"__data_standard_{self.table}__", self.__data_standard__)
        df = func(df, **kwargs)
        return df
    
    
    
   
    
    
