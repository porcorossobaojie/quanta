# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:30:31 2026

@author: Porco Rosso
"""
from typing import Any
import numpy as np
import pandas as pd
import jqdatasdk as jq

from ....libs.utils import calendar
from ....libs.db.main import main as db
from ._common import common
from ....config import settings

config = settings('data')

class main(common, type('recommand_settings', (), config.public_keys.minfreq_settings.key), db):
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
    def trade_days(self) -> pd.DatetimeIndex:
        """Retrieves the trading days from the internal calendar | 获取内部日历的交易日期"""
        return self.calendar.trade_days
    
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
    