# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:33:11 2026

@author: Porco Rosso
"""

import jqdatasdk as jq
from typing import Any, Literal, Optional, Union, Dict
import pandas as pd

from quanta.data.joinquant.meta.main import main as meta
from quanta.config import settings

config = settings('data')


class main(meta):
    """
    ===========================================================================
    Main class for handling trade date table data from JoinQuant.

    This class extends the meta class to provide specific data processing
    and daily update functionalities for trade date related tables.
    ---------------------------------------------------------------------------
    处理 JoinQuant 交易日期表数据的主类.

    此类扩展了元类, 为交易日期相关表提供特定的数据处理和每日更新功能.
    ---------------------------------------------------------------------------
    """

    def __data_standard_aindexweights__(
        self,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Specialized standardization hook for index weight data. It leverages
        the base class standardization and then applies specific time bias
        and weight scaling (percentage to decimal).

        Parameters
        ----------
        df : pd.DataFrame
            The input weight data.
        **kwargs : Any
            Additional arguments including 'start_date'.

        Returns
        -------
        pd.DataFrame
            Standardized weight data.
        -----------------------------------------------------------------------
        指数权重数据的专用标准化钩子. 它利用基类标准化, 然后应用特定的时间偏移和
        权重缩放 (百分比转小数).

        参数
        ----
        df : pd.DataFrame
            输入的权重数据.
        **kwargs : Any
            包括 'start_date' 在内的附加参数.

        返回
        ----
        pd.DataFrame
            标准化后的权重数据.
        -----------------------------------------------------------------------
        """
        df = self.__data_standard__(df, **kwargs)
        df[self.trade_dt] = pd.to_datetime(kwargs.get('start_date')) + pd.Timedelta(pd.Timedelta(config.tables.recommand_settings.time_bias))
        df[list(self.columns_information.get('weight').keys())[0]] = df[list(self.columns_information.get('weight').keys())[0]] / 100
        return df

    def __data_standard_astockindustrys__(
        self,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Specialized standardization hook for stock industry data. It cleans
        and reformats industry-specific column names before calling base
        standardization.

        Parameters
        ----------
        df : pd.DataFrame
            The input industry data.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            Standardized industry data.
        -----------------------------------------------------------------------
        股票行业数据的专用标准化钩子. 在调用基类标准化之前, 它会清理并重新格式化
        行业特定的列名.

        参数
        ----
        df : pd.DataFrame
            输入的行业数据.
        **kwargs : Any
            附加参数.

        返回
        ----
        pd.DataFrame
            标准化后的行业数据.
        -----------------------------------------------------------------------
        """
        df.columns = ['_'.join([''.join(i[0].split('_')), i[1].split('_')[-1]]) for i in df.columns]
        df = self.__data_standard__(df, **kwargs)
        return df

    def __data_standard_astockconcept__(
        self,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Specialized standardization hook for stock concept data. It injects
        the 'start_date' as a level indicator for tracking concept changes.

        Parameters
        ----------
        df : pd.DataFrame
            The input concept data.
        **kwargs : Any
            Additional arguments including 'start_date'.

        Returns
        -------
        pd.DataFrame
            Standardized concept data.
        -----------------------------------------------------------------------
        股票概念数据的专用标准化钩子. 它注入 'start_date' 作为层级指示符, 用于
        跟踪概念变化.

        参数
        ----
        df : pd.DataFrame
            输入的概念数据.
        **kwargs : Any
            包括 'start_date' 在内的附加参数.

        返回
        ----
        pd.DataFrame
            标准化后的概念数据.
        -----------------------------------------------------------------------
        """
        df['level_1'] = kwargs.get('start_date')
        df = self.__data_standard__(df, **kwargs)
        return df

    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Overrides the base pipeline to include automatic return calculation.
        It uses (adjusted) close prices to derive returns and filters out rows
        containing only metadata.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for the data extraction process.

        Returns
        -------
        pd.DataFrame
            The fully processed and augmented DataFrame.
        -----------------------------------------------------------------------
        重写基类流水线以包含自动收益率计算. 它使用 (复权) 收盘价来推导收益率, 并
        过滤掉仅包含元数据的行.

        参数
        ----
        **kwargs : Any
            数据提取过程的关键字参数.

        返回
        ----
        pd.DataFrame
            完全处理和增强后的 DataFrame.
        -----------------------------------------------------------------------
        """
        df = super().pipeline(**kwargs)

        # construct returns of portfolio by calculated with close price(first use adj price if have)
        if isinstance(self.columns_information, dict):
            ret_key = self.columns_information.get('returns', None)
        else:
            x = eval(f"jq.get_table_info({self.columns_information})")
            x.iloc[:, 0] = x.iloc[:, 0].replace(config.tables.transform | {'code': self.code})
            x.iloc[:, 2] = x.iloc[:, 2].replace({'date': 'datetime', 'DATE': 'datetime'})
            x = x.set_index(x.columns[0]).iloc[:, [1, 0]].T.to_dict('list')
            ret_key = x.get('returns', None)
        if ret_key is not None:
            ret_key = list(ret_key.keys())[0]
            try:
                df[ret_key] = df['close_adj'] / df['preclose_adj'] - 1
            except Exception:
                df[ret_key] = df['close'] / df['preclose'] - 1
        df = df[df.drop([self.trade_dt, self.code], axis=1, errors='ignore').notnull().any(axis=1)]
        return df

    def daily(self, if_exists: Literal['append', 'replace'] = 'append') -> None:
        """
        =======================================================================
        Performs daily updates for trade-date based tables. Special handling
        is applied to 'astocklisting' which is fully refreshed daily. For
        others, it performs incremental updates based on the last trade date.

        Parameters
        ----------
        if_exists : Literal['append', 'replace']
            Strategy when the table already exists. Default is 'append'.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        执行基于交易日期的表的每日更新. 对每日全量刷新的 'astocklisting' 进行
        特殊处理. 对于其他表, 根据最后一个交易日执行增量更新.

        参数
        ----
        if_exists : Literal['append', 'replace']
            当表已存在时的策略. 默认为 'append'.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        if self.table == 'astocklisting':  # this table inform the on list time for each stock, which will replace every day
            self.drop_table()
            self.create_table()
            df = self.pipeline()
            self.__write__(df, log=True)

        else:
            if if_exists == 'replace':
                self.drop_table()

            if not self.table_exist():
                getattr(self, f"create_table_{self.table}", self.create_table)()

            id_key = self.__find_max_of_exist_table__(self.trade_dt)
            days = self._trade_days[self._trade_days > id_key]

            if len(days):
                i = 0
                periods = self.periods
                total = len(days)

                while i < total or jq.get_query_count()['spare'] < 1000000:
                    start_date = days[i]
                    end_idx = min(i + periods - 1, total - 1)
                    end_date = days[end_idx]
                    df = self.pipeline(start_date=str(start_date.date()), end_date=str(end_date.date()))
                    print(f"Update to date: {end_date}; Query count left: {jq.get_query_count()['spare']}")
                    self.__write__(df, log=True)
                    if end_date >= days.max():
                        break
                    i = end_idx + 1
