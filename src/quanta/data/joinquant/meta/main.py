# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:43:22 2026

@author: Porco Rosso
"""

from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import jqdatasdk as jq

from quanta.libs.utils import merge_dicts
from quanta.config import settings, login_info
from quanta.libs.db.main import main as db

config = settings('data')


class main(db, type('recommand_settings', (), config.tables.recommand_settings.key)):
    """
    ===========================================================================
    Base metadata and connection class for JoinQuant data extraction,
    providing core data processing, standardization, and table management.
    ---------------------------------------------------------------------------
    用于 JoinQuant 数据提取的基础元数据和连接类, 提供核心数据处理, 标准化和表
    管理功能.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs: Any):
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
        _trade_days = pd.to_datetime(jq.get_trade_days('2005-01-01')) + pd.Timedelta(config.tables.recommand_settings.time_bias)
        self._trade_days = _trade_days[_trade_days <= pd.Timestamp.today() - pd.Timedelta(4, 'h')]

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
        """
        =======================================================================
        Retrieves the standard asset code column name for the current
        portfolio type.

        Returns
        -------
        str
            The code column name.
        -----------------------------------------------------------------------
        获取当前投资组合类型的标准资产代码列名.

        返回
        ----
        str
            代码列名.
        -----------------------------------------------------------------------
        """
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
        # add time bias on trade_dt or ann_dt
        for i in [self.ann_dt, self.trade_dt]:
            if i in df.columns:
                df[i] = pd.to_datetime(df[i]) + pd.Timedelta(config.tables.recommand_settings.time_bias)
            if (i not in df.columns) and i in self.columns.keys():
                try:
                    df[i] = pd.to_datetime(kwargs['start_date']) + pd.Timedelta(config.tables.recommand_settings.time_bias)
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
                id_key = pd.to_datetime(getattr(self, 'date_start', config.tables.recommand_settings.date_start))
            else:
                id_key = 0
        return id_key

    def table_exist(self) -> bool:
        """
        =======================================================================
        Checks if the current table exists in the database.

        Returns
        -------
        bool
            True if the table exists, False otherwise.
        -----------------------------------------------------------------------
        检查数据库中是否存在当前表.

        返回
        ----
        bool
            如果表存在则为 True, 否则为 False.
        -----------------------------------------------------------------------
        """
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
            partition = None if keys != self.trade_dt else {self.trade_dt: eval(config.tables.recommand_settings.key.partition)}
            parameters = (
                self.__parameters__()
                | {'keys': keys, 'partition': partition}
                | {'columns': self.columns, 'log': True}
                | kwargs
            )
        else:
            parameters = self.__parameters__(parameters, kwargs)
        super().__create_table__(**parameters)
