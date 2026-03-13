# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 14:51:56 2026

@author: Porco Rosso
"""
import pandas as pd
import numpy as np
from functools import reduce
from functools import lru_cache
from typing import Optional, Union, List, Dict, Any

from quanta.libs._flow._main._connect import main as meta_table, trade_days, calendar_days
from ._connect import main as meta_table, trade_days, calendar_days
from quanta.config import settings

table_info = settings('data').public_keys.recommand_settings
config = settings('flow')

# Global dictionary to map portfolio types to their respective tables
TABLE_DIC = {i: {} for i in table_info.portfolio_types}
TABLES = meta_table.table_info()['table_name'].unique()

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

    def __init__(self, portfolio_type: str = 'astock'):
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
    def _help(self) -> pd.DataFrame:
        """
        =======================================================================
        Returns internal table information filtered by the current
        portfolio type.

        Returns
        -------
        pd.DataFrame
            DataFrame containing metadata for relevant tables.
        -----------------------------------------------------------------------
        返回按当前投资组合类型过滤的内部表信息.

        返回
        ----
        pd.DataFrame
            包含相关表元数据的 DataFrame.
        -----------------------------------------------------------------------
        """
        x = meta_table().table_info()
        x = x[x.iloc[:, -3].str.contains(self.portfolio_type)]
        return x

    def help(self, col: str) -> pd.DataFrame:
        """
        =======================================================================
        Finds table and column information for a given column name.

        Parameters
        ----------
        col : str
            The column name to search for.

        Returns
        -------
        pd.DataFrame
            Information about which tables contain the specified column.
        -----------------------------------------------------------------------
        查找给定列名的表和列信息.

        参数
        ----
        col : str
            要搜索的列名.

        返回
        ----
        pd.DataFrame
            关于哪些表包含指定列的信息.
        -----------------------------------------------------------------------
        """
        return meta_table.__find__(col, self._help)

    def __columns_to_tables__(
        self,
        columns: Union[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        =======================================================================
        Maps a list of column names to their respective source tables.

        Parameters
        ----------
        columns : Union[str, List[str]]
            Single column name or list of column names.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where keys are table names and values are lists of
            columns belonging to those tables.
        -----------------------------------------------------------------------
        将列名列表映射到其各自的源表.

        参数
        ----
        columns : Union[str, List[str]]
            单个列名或列名列表.

        返回
        ----
        Dict[str, List[str]]
            一个字典, 键为表名, 值为属于这些表的列列表.
        -----------------------------------------------------------------------
        """
        tables = self._help
        if isinstance(columns, str) and len(columns.split('-')) > 1:
            index = columns.split('-')
            index = tables[tables.iloc[:, -3].str.contains(index[0]) & (tables.iloc[:, -2] == index[1])]
        else:
            columns = [columns] if isinstance(columns, str) else [i for i in columns]
            index = [tables[tables.iloc[:, -2] == i].index for i in columns]
            index = index[0].append(index[1:])
            index = tables.loc[index]
        index = index.groupby(index.columns[-3])[index.columns[-2]].apply(list).to_dict()
        return index

    def __call__(
        self,
        columns: Union[str, List[Any]],
        end: Optional[pd.Timestamp] = None,
        quarter_adj: bool = False,
        quarter_diff: bool = False,
        shift: int = 0,
        **kwargs: Any
        ) -> Union[pd.Series, pd.DataFrame]:
        """
        =======================================================================
        Retrieves data for specified columns across multiple tables.

        Parameters
        ----------
        *columns : str
            Variable number of column names to fetch.
        end : Optional[pd.Timestamp]
            End date for time-based filtering. Default is None.
        quarter_adj : bool
            Whether to apply financial quarter adjustment. Default is False.
        quarter_diff : bool
            Indicator for quarter difference calculation. Default is False.
        shift : int
            Number of periods to shift the data. Default is 0.
        **kwargs : Any
            Additional keyword arguments.
        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            The fetched data, concatenated and formatted.
        -----------------------------------------------------------------------
        跨多个表检索指定列的数据.

        参数
        ----
        columns : Union[str, List[Any]]
            要获取的列名.
        end : Optional[pd.Timestamp]
            基于时间的过滤的结束日期. 默认为 None.
        quarter_adj : bool
            是否应用财务季度调整. 默认为 False.
        quarter_diff : bool
            季度差计算指示器. 默认为 False.
        shift : int
            数据位移周期. 默认为 0.
        返回
        ----
        Union[pd.Series, pd.DataFrame]
            获取的数据, 经过合并和格式化.
        -----------------------------------------------------------------------
        """
        dic = self.__columns_to_tables__(columns)
        if len(dic) - 1:
            df = {i: getattr(self, i)(
                j, 
                end=end, quarter_adj=quarter_adj, quarter_diff=quarter_diff, shift=shift, **kwargs
                ) for i, j in dic.items()}
            df = pd.concat(df, axis=1)
        else:
            df = [getattr(self, i)(
                j[0] if len(j) == 1 else j, 
                end=end, quarter_adj=quarter_adj, quarter_diff=quarter_diff, shift=shift, **kwargs
                ) for i, j in dic.items()][0]

        if isinstance(df, pd.Series) or (isinstance(df, pd.DataFrame) & (df.columns.nlevels > 1)):
            return df
        else:
            for i in range(0, df.columns.nlevels, -1):
                if df.columns.get_level_values(i).unique() == 1:
                    df.columns = df.colums.droplevel(i)
            return df

    def finance(
        self,
        column: Union[str, List[str]],
        quarter_adj: bool = False,
        quarter_diff: int = 1,
        shift: int = 0,
        periods: int = 1,
        min_periods: Optional[int] = None,
        drop_zero: bool = True,
    ) -> pd.DataFrame:
        """
        =======================================================================
        Accesses financial statement data with specialized adjustments for
        reporting periods and cumulative-to-single-quarter conversion.

        Parameters
        ----------
        column : Union[str, List[str]]
            Financial column(s) to fetch.
        quarter_adj : Union[bool, int]
            If not False, an integer representing the starting month of the
            natural year's reports. This handles Year-to-Date (YTD) cumulative
            data and converts it into single-quarter values (e.g., operating
            profit). Default is False.
        quarter_diff : int
            The number of quarters used for differencing when `quarter_adj` is
            active. This controls the lag for calculating single-quarter increments:
                1. Standard quarterly: quarter_adj=3, quarter_diff=1
                2. Semi-annual only: quarter_adj=6, quarter_diff=2
            Default is 1.
        shift : int
            Periods to shift the data. Default is 0.
        periods : int
            Number of historical reports to look back from the current time
            point. Default is 1.
        min_periods : Optional[int]
            Minimum number of periods required for the rolling window.
            Default is None.
        drop_zero : bool
            Whether to treat zero values as NaNs. This is particularly useful
            when `quarter_adj=6, quarter_diff=2`, as some financial reports use
            0 to fill periods without provided data. Default is True.

        Returns
        -------
        pd.DataFrame
            The processed financial data.
        -----------------------------------------------------------------------
        访问具有特定报告期调整的财务报表数据.

        参数
        ----
        column : Union[str, List[str]]
            要获取的财务列.
        quarter_adj : bool
            是否按报告季度调整数据. 默认为 False,若非False,需给出int,代表自然年的起
            始季报的月份.
            季度调整意味着源数据是按自然年的季报累加的,希望通过这个参数把数据转化成单
            季度的值(e.g, opera porfit).
        quarter_diff : int
            用于计算的季度差. 默认为 1.
            当quanta_adj非False时,这个参数才有意义.
            通过这个参数,来控制从累加差分获得的单季度的差分对应期数,因为在财报中会有
            不同的更新方式:
                1. 按季报更新数据,通常: quanter_adj = 3, quarter_diff = 1
                2. 按半年报更新数据,则: quanter_adj = 6, quanter_diff = 2
        shift : int
            数据位移周期. 默认为 0.
        periods : int
            以当前时间节点,往前取n期的财报数据, 默认为 1.
        min_periods : Optional[int]
            滚动窗口的最小周期. 默认为 None.
        drop_zero : bool
            是否将零值视为 NaN. 默认为 True.
            这里主要是配合 quanter_adj = 6, quanter_diff = 2的情况,有些财报中会用0
            对这期报告中不提供的数据进行填充

        返回
        ----
        pd.DataFrame
            处理后的财务数据.
        -----------------------------------------------------------------------
        """
        dic = self.__columns_to_tables__(column)
        df = getattr(self, list(dic.keys())[0]).__finance__(list(dic.values())[0][0], quarter_adj, quarter_diff, shift, periods, min_periods, drop_zero)
        return df

    @lru_cache(maxsize=8)
    def multilize(
        self,
        column: str,
        value_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        =======================================================================
        Reshapes long-format data into a wide-format cross-sectional
        DataFrame.

        Parameters
        ----------
        column : str
            The column used as a pivot index/category.
        value_column : Optional[str]
            The column containing values to populate the wide DataFrame.
            If None, returns boolean indicators of presence. Default is None.

        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame indexed by date/asset.
        -----------------------------------------------------------------------
        将长格式数据重塑为宽格式横截面 DataFrame.

        参数
        ----
        column : str
            用作透视索引/类别的列.
        value_column : Optional[str]
            包含用于填充宽格式 DataFrame 的值的列.
            如果为 None, 则返回是否存在布尔指示. 默认为 None.

        返回
        ----
        pd.DataFrame
            按日期/资产索引的宽格式 DataFrame.
        -----------------------------------------------------------------------
        """
        if value_column is None:
            df = self(column)
            if not df.index.duplicated().any():
                df = df.stack()
            df = df.to_frame(column)
            df['temp_value'] = 1
        else:
            df = self(column, value_column)
        df = df.set_index(column, append=True)
        df = df.iloc[:, 0].unstack(0).T
        if df.columns.names[0] in [table_info.key.astock_code, table_info.key.afund_code]:
            df.columns = df.columns.swaplevel(-1, 0)
        if value_column is None:
            df = df.notnull()
        df = df.sort_index(axis=1).sort_index()
        return df

    @lru_cache(maxsize=8)
    def _astock_listing(self, limit: int = 126) -> pd.DataFrame:
        """
        =======================================================================
        Calculates A-stock listing status based on listing dates and a minimum
        listing duration.

        Parameters
        ----------
        limit : int
            Minimum number of trading days since listing. Default is 126.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating listing status.
        -----------------------------------------------------------------------
        根据上市日期和最小上市时长计算 A 股上市状态.

        参数
        ----
        limit : int
            自上市以来的最小交易日数. 默认为 126.

        返回
        ----
        pd.DataFrame
            指示上市状态的布尔值 DataFrame.
        -----------------------------------------------------------------------
        """
        if not hasattr(self, '_internal_listing'):
            df = (self([config.listing.astock_listing_date, config.listing.astock_delisting_date]).clip(
                upper=pd.to_datetime(pd.Timestamp.today().date()))
                .set_index(config.listing.astock_delisting_date, append=True)[config.listing.astock_listing_date]).unstack(0)
            df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='d')).bfill().dropna(how='all', axis=1)
            df.index = df.index + pd.Timedelta(meta_table.time_bias)
            df = df.reindex(trade_days)
            x = df[df.sub(df.index, axis=0).astype('int64') <= 0].notnull().cumsum()
            setattr(self, '_internal_listing', x)
        x = getattr(self, '_internal_listing')
        x = x >= limit
        return x

    @lru_cache(maxsize=8)
    def _afund_listing(self, limit: int = 126) -> pd.DataFrame:
        """
        =======================================================================
        Calculates fund listing status based on listing history.

        Parameters
        ----------
        limit : int
            Minimum number of periods since listing. Default is 126.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating listing status.
        -----------------------------------------------------------------------
        根据上市历史计算基金上市状态.

        参数
        ----
        limit : int
            自上市以来的最小周期数. 默认为 126.

        返回
        ----
        pd.DataFrame
            指示上市状态的布尔值 DataFrame.
        -----------------------------------------------------------------------
        """
        if not hasattr(self, '_internal_listing'):
            x = self(config.listing.afund_listing_date).notnull().cumsum()
            setattr(self, '_internal_listing', x)
        x = getattr(self, '_internal_listing')
        x = x >= limit
        return x

    def listing(self, limit: int = 126) -> pd.DataFrame:
        """
        =======================================================================
        Interface to get listing status for the current portfolio type.

        Parameters
        ----------
        limit : int
            Minimum duration since listing. Default is 126.

        Returns
        -------
        pd.DataFrame
            Boolean mask of listed assets.
        -----------------------------------------------------------------------
        获取当前投资组合类型上市状态的接口.

        参数
        ----
        limit : int
            自上市以来的最小持续时间. 默认为 126.

        返回
        ----
        pd.DataFrame
            上市资产的布尔掩码.
        -----------------------------------------------------------------------
        """
        func = getattr(self, f"_{self.portfolio_type}_listing")
        df = func(limit)
        return df

    @lru_cache(maxsize=1)
    def not_st(self, value: int = 1) -> pd.DataFrame:
        """
        =======================================================================
        Generates a boolean mask indicating assets that are not under
        special treatment (ST).

        Parameters
        ----------
        value : int
            Threshold for ST status codes. Default is 1.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame where True indicates non-ST status.
        -----------------------------------------------------------------------
        生成指示非特别处理 (ST) 资产的布尔掩码.

        参数
        ----
        value : int
            ST 状态码的阈值. 默认为 1.

        返回
        ----
        pd.DataFrame
            True 表示非 ST 状态的布尔值 DataFrame.
        -----------------------------------------------------------------------
        """
        key = config.status.not_st
        dic = self.__columns_to_tables__(key)
        table_obj = [getattr(self, i) for i, j in dic.items()][0]
        df = table_obj.__read__().set_index(table_obj.index_keys)[key]
        status = {301001: 0, 301002: 1, 301003: 2, 301005: 3, 301006: 4}
        df = df.replace(status)
        df = df[df.isin(status.values())]
        df = df.loc[~df.index.duplicated(keep='last')].unstack(table_obj.code).sort_index(axis=1).sort_index()
        df = df.ffill().reindex(calendar_days).ffill().reindex(trade_days).loc[meta_table.start_date:]
        df = df < value
        return df

    @lru_cache(maxsize=1)
    def traced_index(self, column: str = 'traced_index_name') -> pd.DataFrame:
        """
        =======================================================================
        Fetches and aligns index tracking information.

        Parameters
        ----------
        column : str
            The column name representing the traced index.
            Default is 'traced_index_name'.

        Returns
        -------
        pd.DataFrame
            Aligned index tracking data.
        -----------------------------------------------------------------------
        获取并对齐指数跟踪信息.

        参数
        ----
        column : str
            表示跟踪指数的列名. 默认为 'traced_index_name'.

        返回
        ----
        pd.DataFrame
            对齐后的指数跟踪数据.
        -----------------------------------------------------------------------
        """
        df = self('end_date', column)
        df['end_date'] = (pd.to_datetime(df['end_date']) + meta_table.time_bias).fillna(trade_days[-1]).dropna()
        x = df.reset_index().drop_duplicates(keep='first', subset=df.index.name).dropna()
        x = x.pivot(index='end_date', columns=df.index.name, values=column).reindex(trade_days)
        x = x.loc[meta_table.start_date:].bfill()
        return x
