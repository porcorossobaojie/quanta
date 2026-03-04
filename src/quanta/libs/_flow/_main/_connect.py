# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:11:56 2026

@author: Porco Rosso
"""

from functools import lru_cache
from typing import Literal, Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from quanta.libs.db.main import main as db
from quanta.config import settings, login_info

config = settings('data').public_keys
columns_info = config.recommand_settings.key


class main(db, type('public_keys', (), config.recommand_settings.key)):
    """
    ===========================================================================
    A specialized database connection class for financial data flow,
    inheriting from the base database class and dynamic public keys.
    ---------------------------------------------------------------------------
    用于金融数据流的专用数据库连接类, 继承自基础数据库类和动态公共键.
    ---------------------------------------------------------------------------
    """
    time_bias = pd.Timedelta(config.recommand_settings.time_bias)
    start_date = pd.to_datetime(settings('flow').start_date) + pd.Timedelta(config.recommand_settings.time_bias)

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
        keys = [i for i in [self.trade_dt, self.ann_dt, self.report_period] if i in self.columns] + [self.code]
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
        returns: bool = False,
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
        if not hasattr(self, '_internal_data'):
            if self.trade_dt in self.filter_key or self.ann_dt in self.filter_key:
                filter_value = self.start_date if self.filter_key == self.trade_dt else self.start_date + pd.offsets.YearEnd(-4)
                where = f"{self.filter_key} >='{filter_value}'"
            else:
                where = kwargs.get('where', None)
            df = self.__read__(where=where)
            df.columns = pd.CategoricalIndex(df.columns)
            df[self.code] = pd.CategoricalIndex(df[self.code])
            df = df.set_index(self.index_keys)

            if self.filter_key == self.trade_dt:
                try:
                    columns = [i for i in self.columns if i not in [self.trade_dt, self.ann_dt, self.report_period, self.code]]
                    df = df.unstack()
                    df = {i: df[i] for i in columns}
                except Exception:
                    print(f"WARNING: UNSTANDARD LOADING ON DATA SOURCE <{self.table}>")
            self._internal_data = df
        if returns:
            return self._internal_data
        return None

    def __read_from_internal__(
        self,
        columns: Union[str, List[Any]]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        =======================================================================
        Extracts specific columns from the internal data cache.

        Parameters
        ----------
        columns : Union[str, List[Any]]
            Column names to extract.

        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            The extracted data slice.
        -----------------------------------------------------------------------
        从内部数据缓存中提取特定列.

        参数
        ----
        columns : Union[str, List[Any]]
            要提取的列名.

        返回
        ----
        Union[pd.Series, pd.DataFrame]
            提取的数据切片.
        -----------------------------------------------------------------------
        """
        adj = True if isinstance(columns, str) else False
        self.__read_from_db__()
        columns = self.__columns_standard__(columns)
        if self.filter_key == self.trade_dt:
            if not adj:
                df = pd.concat({i: getattr(self, '_internal_data')[i] for i in columns}, axis=1)
                df.columns.names = ['value_key'] + list(df.columns.names)[1:]
            else:
                df = getattr(self, '_internal_data')[columns[0]]
        else:
            df = getattr(self, '_internal_data')[columns]
            if adj:
                df = df.iloc[:, 0]
        return df

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
        Primary interface to fetch and process data from the table.

        Parameters
        ----------
        columns : Union[str, List[Any]]
            Column names to fetch.
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
            The processed and aligned data.
        -----------------------------------------------------------------------
        从表中获取并处理数据的主要接口.

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
        **kwargs : Any
            附加关键字参数.

        返回
        ----
        Union[pd.Series, pd.DataFrame]
            处理并对齐后的数据.
        -----------------------------------------------------------------------
        """
        df = self.__read_from_internal__(columns, **kwargs)
        if end is not None and len(set([self.trade_dt, self.ann_dt]) & set(self.columns)):
            df = df[df.index.get_level_values(self.filter_key) <= end]

        if self.ann_dt in df.index.names:
            df.index = df.index.droplevel(self.ann_dt)
            try:
                df = df.unstack()
                filter_end = pd.Timestamp.today() if end is None else end
                df = df.reindex(pd.date_range(df.index.min(), filter_end, freq='QE', name=df.index.name))
                df = df.sort_index(axis=1).sort_index()
                if quarter_adj:
                    df = self.__finance_quarter_adjust__(df, quarter_adj, quarter_diff)
                if shift:
                    df = self.__finance_shift__(df, shift)
            except Exception:
                pass
        elif self.trade_dt in df.index.names:
            df = df.loc[self.start_date:]
        return df

    def __finance_shift__(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        =======================================================================
        Conditionally shifts columns in financial data based on NaN values in
        the last row.

        Parameters
        ----------
        df : pd.DataFrame
            The financial DataFrame to shift.
        n : int
            Maximum number of shifts.

        Returns
        -------
        pd.DataFrame
            The shifted DataFrame.
        -----------------------------------------------------------------------
        根据最后一行中的 NaN 值有条件地移动财务数据中的列.

        参数
        ----
        df : pd.DataFrame
            要移动的财务 DataFrame.
        n : int
            最大移动次数.

        返回
        ----
        pd.DataFrame
            移动后的 DataFrame.
        -----------------------------------------------------------------------
        """
        bools = df.iloc[-1].isnull()
        while n > 0 and bools.any():
            n -= 1
            df.loc[:, bools] = df.loc[:, bools].shift()
            bools = df.iloc[-1].isnull()
        return df

    def __finance_quarter_adjust__(
        self,
        df: pd.DataFrame,
        month: int,
        quarter_diff: int
    ) -> pd.DataFrame:
        """
        =======================================================================
        Adjusts financial data based on reporting quarters.

        Parameters
        ----------
        df : pd.DataFrame
            The input financial DataFrame.
        month : int
            The target reporting month for adjustment.
        quarter_diff : int
            The number of quarters to difference.

        Returns
        -------
        pd.DataFrame
            The quarter-adjusted DataFrame.
        -----------------------------------------------------------------------
        根据报告季度调整财务数据.

        参数
        ----
        df : pd.DataFrame
            输入财务 DataFrame.
        month : int
            调整的目标报告月份.
        quarter_diff : int
            差分季度数.

        返回
        ----
        pd.DataFrame
            季度调整后的 DataFrame.
        -----------------------------------------------------------------------
        """
        if self.report_period in df.index.names:
            day = 31 if month in [1, 3, 5, 7, 8, 10, 12] else (30 if month in [4, 6, 9, 11] else 28)
            index = df.index.get_level_values(self.report_period)
            tmp = df[(index.month == month) & (index.day == day)]
            df = df.diff(quarter_diff)
            df.loc[tmp.index] = tmp
        return df

    def __finance_periods_merge__(
        self,
        df: Union[pd.Series, pd.DataFrame],
        periods: int,
        quarter_adj: bool = False,
        diff: int = 1,
        drop_zero: bool = False
    ) -> pd.DataFrame:
        """
        =======================================================================
        Merges financial data across multiple reporting periods.

        Parameters
        ----------
        df : Union[pd.Series, pd.DataFrame]
            The input financial data.
        periods : int
            Number of periods to merge.
        quarter_adj : bool
            Whether to apply quarter-based adjustment before merging.
            Default is False.
        diff : int
            Quarter difference for adjustment. Default is 1.
        drop_zero : bool
            Placeholder for zero-dropping logic. Default is False.

        Returns
        -------
        pd.DataFrame
            The merged financial DataFrame.
        -----------------------------------------------------------------------
        合并多个报告期的财务数据.

        参数
        ----
        df : Union[pd.Series, pd.DataFrame]
            输入财务数据.
        periods : int
            要合并的周期数.
        quarter_adj : bool
            合并前是否应用基于季度的调整. 默认为 False.
        diff : int
            用于调整的季度差. 默认为 1.
        drop_zero : bool
            零值删除逻辑的占位符. 默认为 False.

        返回
        ----
        pd.DataFrame
            合并后的财务 DataFrame.
        -----------------------------------------------------------------------
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.columns = [0]
        df = df.sort_index().reset_index()
        if quarter_adj:
            df = df.sort_values([self.code, self.report_period])
            df['tmp_year'] = df[self.report_period].dt.year
            df['tmp'] = df.groupby(['tmp_year', self.code])[0].diff(diff).fillna(df[0])
            df = df[[self.ann_dt, self.report_period, self.code, 'tmp']].rename({'tmp': 0}, axis=1)

        def next_period(sub_df, shift_val):
            ne = sub_df.copy()
            ne.columns = ne.columns[:-1].to_list() + [shift_val * -1]
            ne[self.report_period] = pd.DatetimeIndex(ne[self.report_period]).shift(shift_val, freq='QE')
            ne = ne.drop(self.ann_dt, errors='ignore', axis=1).set_index([self.report_period, self.code])
            return ne

        merges = [next_period(df, i) for i in range(1, periods)]
        if len(merges):
            merges = pd.concat(merges, axis=1)
            x = pd.merge(df, merges.reset_index(), on=[self.report_period, self.code], how='left')
        else:
            x = df
        x = x.drop_duplicates(subset=[self.ann_dt, self.code], keep='last')
        return x

    @lru_cache(maxsize=8)
    def __finance__(
        self,
        column: str,
        quarter_adj: bool = False,
        quarter_diff: int = 1,
        shift: int = 0,
        periods: int = 1,
        min_periods: Optional[int] = None,
        drop_zero: bool = True,
        day_index: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        =======================================================================
        Comprehensive financial data processing, including merging periods,
        adjusting for report lag, and reindexing to trade days.

        Parameters
        ----------
        column : str
            Financial column to process.
        quarter_adj : bool
            Whether to apply quarter adjustment. Default is False.
        quarter_diff : int
            Difference for quarter adjustment. Default is 1.
        shift : int
            Number of periods to shift reporting. Default is 0.
        periods : int
            Number of reporting periods to combine. Default is 1.
        min_periods : Optional[int]
            Minimum periods required for results. Default is None.
        drop_zero : bool
            Whether to treat zero values as NaNs. Default is True.
        day_index : Optional[pd.DatetimeIndex]
            Custom index for reindexing. Defaults to calendar_days.

        Returns
        -------
        pd.DataFrame
            Fully processed financial DataFrame.
        -----------------------------------------------------------------------
        综合财务数据处理, 包括合并周期, 调整报告滞后以及对齐到交易日.

        参数
        ----
        column : str
            要处理的财务列.
        quarter_adj : bool
            是否应用季度调整. 默认为 False.
        quarter_diff : int
            季度调整的差值. 默认为 1.
        shift : int
            报告位移周期. 默认为 0.
        periods : int
            要组合的报告周期数. 默认为 1.
        min_periods : Optional[int]
            结果所需的最小周期数. 默认为 None.
        drop_zero : bool
            是否将零值视为 NaN. 默认为 True.
        day_index : Optional[pd.DatetimeIndex]
            用于重索引的自定义索引. 默认为 calendar_days.

        返回
        ----
        pd.DataFrame
            完全处理后的财务 DataFrame.
        -----------------------------------------------------------------------
        """
        day_index = calendar_days if day_index is None else day_index
        df = self.__read_from_internal__(column)
        if drop_zero:
            df = df[df != 0]
        x = self.__finance_periods_merge__(df, periods, quarter_adj, quarter_diff)
        x = x.set_index(self.ann_dt).sort_index()
        x.index = x.index.astype('datetime64[ns]')
        fill_index = pd.MultiIndex.from_product([day_index, x[self.code].unique()], names=[self.trade_dt, self.code])
        obj = pd.DataFrame(index=fill_index).reset_index(self.code)
        obj = pd.merge_asof(obj, x, by=self.code, left_index=True, right_index=True)

        obj['Q'] = obj.index.to_period('Q')
        obj['Q'] = obj['Q'].dt.year * 4 + obj['Q'].dt.quarter
        obj[self.report_period] = obj[self.report_period].dt.to_period('Q')
        obj[self.report_period] = obj[self.report_period].dt.year * 4 + obj[self.report_period].dt.quarter
        obj['Q'] = obj['Q'] - obj[self.report_period] - 1 - shift
        obj = obj.set_index([self.code, 'Q'], append=True).drop(self.report_period, axis=1).dropna(how='all')
        meta = obj[obj.index.get_level_values('Q') <= 0]
        obj = obj[~obj.index.isin(meta.index)]
        if len(obj):
            while (obj.index.get_level_values('Q') > 0).any():
                obj[obj.index.get_level_values('Q') > 0] = obj[obj.index.get_level_values('Q') > 0].shift(axis=1)
                obj = obj.reset_index('Q')
                obj['Q'] -= 1
                obj = obj.set_index('Q', append=True)
                obj = obj.dropna(how='all')
            obj = pd.concat([meta, obj])
        else:
            obj = meta

        obj = obj.reset_index('Q').drop('Q', axis=1)
        obj.columns.name = self.report_period
        obj = obj.unstack(self.trade_dt)
        obj = obj.swaplevel(self.trade_dt, self.report_period, axis=1).T
        if len(obj.index.get_level_values(self.report_period).unique()) == 1:
            obj.index = obj.index.get_level_values(self.trade_dt)
        obj = obj.sort_index().sort_index(axis=1)
        if min_periods is not None:
            obj = obj[obj.groupby(self.trade_dt).transform('count') >= min_periods]
        obj = obj[obj.index.get_level_values(0).isin(trade_days)].loc[self.start_date:]
        return obj


calendar_days = pd.date_range(
    start=pd.to_datetime(main.date_start) + main.time_bias,
    freq='d',
    end=pd.Timestamp.today() - pd.Timedelta(4, 'h'))

try:
    import jqdatasdk as jq
    jq.auth(**login_info('account').joinquant)
    trade_days = pd.to_datetime(
        jq.get_trade_days(
            main.date_start,
            pd.Timestamp.today() - pd.Timedelta(4, 'h') - main.time_bias
        )
    ) + main.time_bias
except Exception:
    trade_days = pd.to_datetime(
        sorted(
            main(table='astockeodprices').__read__(columns=main.trade_dt).iloc[:, 0].unique()
        )
    )
