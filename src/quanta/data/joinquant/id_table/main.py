# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:11:28 2026

@author: Porco Rosso
"""

from typing import Any, Literal, Optional, Dict
import pandas as pd

from quanta.data.joinquant.meta.main import main as meta
from quanta.config import settings

config = settings('data')


class main(meta):
    """
    ===========================================================================
    Main class for handling announcement date table data from JoinQuant.

    This class extends the meta class to provide specific data processing
    and daily update functionalities for announcement date related tables.
    ---------------------------------------------------------------------------
    处理 JoinQuant 公告日期表数据的主类.

    此类扩展了元类, 为公告日期相关表提供特定的数据处理和每日更新功能.
    ---------------------------------------------------------------------------
    """

    def create_table_afundshare(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Specialized table creation for 'afundshare', mapping report periods
        to trade dates in the schema.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for table creation.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        'afundshare' 的专用表创建方法, 在模式中将报告期映射到交易日期.

        参数
        ----
        **kwargs : Any
            表创建的附加参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        columns = {self.trade_dt if i == self.report_period else i: j for i, j in self.columns.items()}
        self.create_table(columns=columns)

    def __data_standard_afundshare__(
        self,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Standardizes 'afundshare' data by mapping report periods to trade
        dates and applying time bias.

        Parameters
        ----------
        df : pd.DataFrame
            The input fund share data.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            Standardized fund share data.
        -----------------------------------------------------------------------
        通过将报告期映射到交易日期并应用时间偏移来标准化 'afundshare' 数据.

        参数
        ----
        df : pd.DataFrame
            输入的基金份额数据.
        **kwargs : Any
            附加参数.

        返回
        ----
        pd.DataFrame
            标准化后的基金份额数据.
        -----------------------------------------------------------------------
        """
        df = self.__data_standard__(df, date=self.report_period)
        df = df.rename({self.report_period: self.trade_dt}, axis=1)
        df[self.trade_dt] = pd.to_datetime(df[self.trade_dt]) + pd.Timedelta(config.tables.recommand_settings.time_bias)
        return df

    def daily(self, if_exists: Literal['append', 'replace'] = 'append') -> None:
        """
        =======================================================================
        Performs daily incremental updates for the announcement date table
        using the primary ID key.

        Parameters
        ----------
        if_exists : Literal['append', 'replace']
            Determines how to handle existing data. 'append' adds new data,
            'replace' drops the table and recreates it. Default is 'append'.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        使用主 ID 键执行公告日期表的每日增量更新.

        参数
        ----
        if_exists : Literal['append', 'replace']
            确定如何处理现有数据. 'append' 添加新数据, 'replace' 删除并重新
            创建表格. 默认为 'append'.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        if if_exists == 'replace':
            self.drop_table()

        if not self.table_exist():
            getattr(self, f"create_table_{self.table}", self.create_table)()

        id_key = self.__find_max_of_exist_table__(self.id_key)
        df = self.pipeline(id_key=id_key)
        self.__write__(df, log=True)
        while len(df):
            id_key = df[self.id_key].max()
            df = self.pipeline(id_key=id_key)
            self.__write__(df, log=True)
