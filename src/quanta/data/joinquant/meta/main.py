# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:43:22 2026

@author: Porco Rosso
"""

from typing import Any, Union
import numpy as np
import pandas as pd
import jqdatasdk as jq
from ....libs.db.main import main as db
from ._common import common
from ....config import settings

config = settings('data')


class main(common, db, type('recommend_settings', (), config.tables.recommend_settings.key)):
    """
    ===========================================================================
    Base metadata and connection class for JoinQuant data extraction,
    providing core data processing, standardization, and table management.
    ---------------------------------------------------------------------------
    用于 JoinQuant 数据提取的基础元数据和连接类, 提供核心数据处理, 标准化和表
    管理功能.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Initializes the meta main instance, setting up the environment and
        fetching security lists.

        Parameters
        ----------
        **kwargs : Any
            Initial configuration and table parameters.
        -----------------------------------------------------------------------
        初始化元主实例, 设置环境并获取证券列表.

        参数
        ----
        **kwargs : Any
            初始配置和表参数.
        -----------------------------------------------------------------------
        """
        super().__init__(**kwargs)
        self.__env_init__()
        self._stock = jq.get_all_securities('stock', date=None).index.tolist()
        _fund = jq.get_all_securities('fund', date=None)
        self._fund = _fund[_fund.iloc[:, -1] == 'etf'].index.tolist()
        self._index = jq.get_all_securities('index', date=None).index.tolist()
        _trade_days = pd.to_datetime(jq.get_trade_days('2005-01-01')) + pd.Timedelta(config.tables.recommend_settings.time_bias)
        self._trade_days = _trade_days[_trade_days <= pd.Timestamp.today() - pd.Timedelta(4, 'h')]

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
        # add time bias on trade_dt or ann_dt
        for i in [self.ann_dt, self.trade_dt]:
            if i in df.columns:
                df[i] = pd.to_datetime(df[i]) + pd.Timedelta(config.tables.recommend_settings.time_bias)
            if (i not in df.columns) and i in self.columns.keys():
                try:
                    df[i] = pd.to_datetime(kwargs['start_date']) + pd.Timedelta(config.tables.recommend_settings.time_bias)
                except KeyError:
                    pass
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


    def __find_max_of_exist_table__(
        self,
        columns: str,
        **kwargs: Any
    ) -> Union[int, float, pd.Timestamp]:
        """
        =======================================================================
        Finds the maximum value of a specific column in an existing table to
        support incremental updates.

        Parameters
        ----------
        columns : str
            The column to find the maximum value for.
        **kwargs : Any
            Additional query arguments.

        Returns
        -------
        Union[int, float, pd.Timestamp]
            The maximum value found, or a default starting value.
        -----------------------------------------------------------------------
        在现有表中查找特定列的最大值, 以支持增量更新.

        参数
        ----
        columns : str
            要查找其最大值的列.
        **kwargs : Any
            附加查询参数.

        返回
        ----
        Union[int, float, pd.Timestamp]
            找到的最大值, 或默认起始值.
        -----------------------------------------------------------------------
        """
        id_key = None
        if self.table_exist():
            id_key = self.__read__(columns=f'MAX({columns})', show_time=False, **kwargs).iloc[0, 0]
            id_key = None if pd.isnull(id_key) else id_key

        if id_key is None:
            if 'DATE' in self.columns.get(columns, ['None'])[0].upper():
                id_key = pd.to_datetime(getattr(self, 'date_start', config.tables.recommend_settings.date_start))
            else:
                id_key = 0
        return id_key

    def table_exist(self) -> bool:
        """Checks if the current table exists in the database | 检查数据库中是否存在当前表"""
        return super().__table_exist__()

    def drop_table(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Drops the current table from the database.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for table dropping.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        从数据库中删除当前表.

        参数
        ----
        **kwargs : Any
            删除表时的附加参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        parameters = self.__parameters__({'log': True}, kwargs)
        super().__drop_table__(**parameters)

    def create_table(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Creates the current table in the database with appropriate schema
        and partitioning if necessary.

        Parameters
        ----------
        **kwargs : Any
            Additional table creation arguments.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        在数据库中创建当前表, 并根据需要设置模式和分区.

        参数
        ----
        **kwargs : Any
            附加建表参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        parameters = {'columns': self.columns, 'log': True}
        if self.engine_type == 'MySQL':
            keys = (
                self.ann_dt
                if self.trade_dt not in self.columns.keys()
                else self.trade_dt
            )
            partition = None if keys != self.trade_dt else {self.trade_dt: eval(config.tables.recommend_settings.key.partition)}
            parameters = (
                self.__parameters__()
                | {'keys': keys, 'partition': partition}
                | {'columns': self.columns, 'log': True}
                | kwargs
            )
        else:
            parameters = self.__parameters__(parameters, kwargs)
        super().__create_table__(**parameters)
