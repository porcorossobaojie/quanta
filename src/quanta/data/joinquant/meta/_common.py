# -*- coding: utf-8 -*-
"""
Shared meta-class hooks for JoinQuant data extraction.

Provides the common attribute and pipeline methods shared by the daily
(`meta/main.py`) and minute-frequency (`meta/mins.py`) table handlers.
--------------------------------------
JoinQuant 数据提取的共享元类钩子.

提供日频 (`meta/main.py`) 与分钟频 (`meta/mins.py`) 表处理器共用的属性
与流水线方法.
"""

from typing import Any, Dict, List
import jqdatasdk as jq
import pandas as pd

from quanta.config import settings
from quanta.libs.utils import merge_dicts

config = settings('data')


class common:
    """
    ===========================================================================
    Mixin providing shared column metadata and pipeline hooks for
    JoinQuant table handlers.
    ---------------------------------------------------------------------------
    为 JoinQuant 表处理器提供共享列元数据与流水线钩子的混入类.
    ---------------------------------------------------------------------------
    """

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
        从当前表名识别投资组合类型.

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
            一个字典, 其中键是列名, 值是列类型.
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
        if df.empty:
            print(f"[quanta] warning: pipeline returned empty data for table <{self.table}>")
        return df
