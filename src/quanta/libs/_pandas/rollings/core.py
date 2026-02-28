# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:19:47 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Callable

from .base import ts_argsort_unit, ts_rank_unit, ts_sort_unit

__all__ = ['_rolls']

class _meta():
    """
    ===========================================================================

    Base class for rolling window operations, providing common utilities.

    ---------------------------------------------------------------------------

    滚动窗口操作的基类，提供通用工具。

    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        df_obj: pd.DataFrame,
        window: int,
        min_periods: int,
        ts_func: Callable,
        ascending: bool
    ):
        """
        ===========================================================================

        Initializes the _meta class with the DataFrame, window, and function details.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input DataFrame for rolling operations.
        window : int
            The size of the rolling window.
        min_periods : int
            Minimum number of observations in window required to have a value.
        ts_func : Callable
            The time-series function to apply within the rolling window.
        ascending : bool
            Whether to sort in ascending order for certain operations.

        ---------------------------------------------------------------------------

        使用DataFrame、窗口和函数详细信息初始化_meta类。

        参数
        ----------
        df_obj : pd.DataFrame
            用于滚动操作的输入DataFrame。
        window : int
            滚动窗口的大小。
        min_periods : int
            窗口中需要有值的最小观测数。
        ts_func : Callable
            在滚动窗口内应用的时间序列函数。
        ascending : bool
            是否按升序排序以进行某些操作。

        ---------------------------------------------------------------------------
        """
        self._obj = df_obj
        self.window = window
        self.min_periods = min_periods
        self._ts_func = ts_func
        self._ascending = ascending

    def __call__(
        self,
        count: Optional[int] = None
    ) -> Any:
        """
        ===========================================================================

        Allows setting the 'count' attribute, typically for specific rolling operations.

        Parameters
        ----------
        count : Optional[int], optional
            The count value to set. Defaults to None.

        Returns
        -------
        Any
            Returns self, allowing for method chaining.

        ---------------------------------------------------------------------------

        允许设置“count”属性，通常用于特定的滚动操作。

        参数
        ----------
        count : Optional[int], optional
            要设置的计数。默认为 None。

        返回
        -------
        Any
            返回自身，允许方法链式调用。

        ---------------------------------------------------------------------------
        """
        if count is not None:
            self._count = count
        return self

    @property
    def count(self) -> int:
        """
        ===========================================================================

        Returns the effective count for rolling operations.

        Returns
        -------
        int
            The count value, defaulting to the window size if not explicitly set.

        ---------------------------------------------------------------------------

        返回滚动操作的有效计数。

        返回
        -------
        int
            计数，如果未明确设置，则默认为窗口大小。

        ---------------------------------------------------------------------------
        """
        x = getattr(self, '_count', self.window)
        return x

    @property
    def _masked_obj(self) -> np.ma.MaskedArray:
        """
        ===========================================================================

        Returns a masked NumPy array of the DataFrame's values, masking NaNs.

        Returns
        -------
        np.ma.MaskedArray
            A masked array where NaN values are masked.

        ---------------------------------------------------------------------------

        返回DataFrame值的掩码NumPy数组，掩盖NaN。

        返回
        -------
        np.ma.MaskedArray
            一个掩码数组，其中NaN值被掩盖。

        ---------------------------------------------------------------------------
        """
        if not hasattr(self, '_masked_obj_'):
            obj = self._obj.values
            self._masked_obj_ = np.ma.array(obj, mask=np.isnan(obj))
        return self._masked_obj_

    @property
    def _min_periods_mask(self) -> pd.DataFrame:
        """
        ===========================================================================

        Generates a mask indicating where the minimum number of periods is met.

        Returns
        -------
        pd.DataFrame
            A DataFrame indicating for each position if the minimum periods condition is met.

        ---------------------------------------------------------------------------

        生成一个掩码，指示满足最小周期数的位置。

        返回
        -------
        pd.DataFrame
            一个DataFrame，指示每个位置是否满足最小周期条件。

        ---------------------------------------------------------------------------
        """
        if not hasattr(self, '_min_periods_mask_'):
            obj = self._obj.notnull().rolling(self.window, min_periods=self.window).sum()
            self._min_periods_mask_ = obj
        return self._min_periods_mask_

    def _rolling_obj(
        self,
        array_obj: np.ndarray,
        pct: Optional[bool],
        group_func: Optional[Callable],
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Performs the core rolling window calculation.

        Parameters
        ----------
        array_obj : np.ndarray
            The NumPy array to apply the rolling window to.
        pct : Optional[bool]
            Whether to return results as percentages. Defaults to None.
        group_func : Optional[Callable]
            The aggregation function to apply within each window. Defaults to None.
        **kwargs : Any
            Additional keyword arguments passed to the `ts_func`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the rolling operation.

        ---------------------------------------------------------------------------

        执行核心滚动窗口计算。

        参数
        ----------
        array_obj : np.ndarray
            要应用滚动窗口的NumPy数组。
        pct : Optional[bool]
            是否以百分比形式返回结果。默认为 None。
        group_func : Optional[Callable]
            应用于每个窗口的聚合函数。默认为 None。
        **kwargs : Any
            传递给 `ts_func` 的附加关键字参数。

        返回
        -------
        pd.DataFrame
            包含滚动操作结果的DataFrame。

        ---------------------------------------------------------------------------
        """
        idx_0, idx_1 = array_obj.shape
        outer = []
        window = self.window
        ascending = self._ascending
        count = self.count * (1 if ascending else -1)
        for i in range(window, idx_0 + 1):
            obj = array_obj[i - window: i]
            obj = self._ts_func(obj, cut=count, pct=pct, func=group_func, **kwargs)
            outer.append(obj)
        df = np.ma.concatenate(outer).reshape(-1, idx_1)
        lens = int(df.shape[0] / (idx_0 - window + 1))
        index = pd.MultiIndex.from_product(
            [self._obj.index[window - 1:], range(lens)],
            names=[self._obj.index.name, 'RANGE']
        ) if lens > 1 else self._obj.index[window - 1:]
        df = pd.DataFrame(df, index=index, columns=self._obj.columns)
        return df

    def _apply(
        self,
        array_obj: np.ndarray,
        group_func: Callable,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Applies a custom function over the rolling window.

        Parameters
        ----------
        array_obj : np.ndarray
            The NumPy array to apply the function to.
        group_func : Callable
            The custom function to apply.
        **kwargs : Any
            Additional keyword arguments passed to `group_func`.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the results of the applied function.

        ---------------------------------------------------------------------------

        在滚动窗口上应用自定义函数。

        参数
        ----------
        array_obj : np.ndarray
            要应用函数的NumPy数组。
        group_func : Callable
            要应用的自定义函数。
        **kwargs : Any
            传递给 `group_func` 的附加关键字参数。

        返回
        -------
        pd.DataFrame
            应用函数结果的DataFrame。

        ---------------------------------------------------------------------------
        """
        return self._rolling_obj(array_obj, pct=None, group_func=group_func, **kwargs)

class _max(_meta):
    """
    ===========================================================================

    A class for calculating rolling maximums and related statistics.

    ---------------------------------------------------------------------------

    用于计算滚动最大值和相关统计数据的类。

    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        df_obj: pd.DataFrame,
        window: int,
        min_periods: int
    ):
        """
        ===========================================================================

        Initializes the _max class, setting up for rolling maximum calculations.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input DataFrame.
        window : int
            The size of the rolling window.
        min_periods : int
            Minimum number of observations in window required to have a value.

        ---------------------------------------------------------------------------

        初始化_max类，为滚动最大值计算做准备。

        参数
        ----------
        df_obj : pd.DataFrame
            输入DataFrame。
        window : int
            滚动窗口的大小。
        min_periods : int
            窗口中需要有值的最小观测数。

        ---------------------------------------------------------------------------
        """
        super().__init__(df_obj, window, min_periods, ts_sort_unit, False)

    def mean(self) -> pd.DataFrame:
        """
        ===========================================================================

        Calculates the rolling mean of the maximum values within each window.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the rolling mean of maximums.

        ---------------------------------------------------------------------------

        计算每个窗口内最大值的滚动平均值。

        返回
        -------
        pd.DataFrame
            包含最大值滚动平均值的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rolling_obj(self._masked_obj, False, np.nanmean)
        x = x.reindex(self._obj.index)
        x = x[self._min_periods_mask >= self.min_periods]
        return x

    def std(self) -> pd.DataFrame:
        """
        ===========================================================================

        Calculates the rolling standard deviation of the maximum values within each window.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the rolling standard deviation of maximums.

        ---------------------------------------------------------------------------

        计算每个窗口内最大值的滚动标准差。

        返回
        -------
        pd.DataFrame
            包含最大值滚动标准差的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rolling_obj(self._masked_obj, False, np.nanstd)
        x = x.reindex(self._obj.index)
        x = x[self._min_periods_mask >= self.min_periods]
        return x

    def sum(self) -> pd.DataFrame:
        """
        ===========================================================================

        Calculates the rolling sum of the maximum values within each window.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the rolling sum of maximums.

        ---------------------------------------------------------------------------

        计算每个窗口内最大值的滚动和。

        返回
        -------
        pd.DataFrame
            包含最大值滚动和的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rolling_obj(self._masked_obj, False, np.nansum)
        x = x.reindex(self._obj.index)
        x = x[self._min_periods_mask >= self.min_periods]
        return x

    def apply(
        self,
        function: Callable,
        **func_kwds: Any
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Applies a custom function to the maximum values within each rolling window.

        Parameters
        ----------
        function : Callable
            The custom function to apply.
        **func_kwds : Any
            Additional keyword arguments to pass to the function.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the results of the applied function.

        ---------------------------------------------------------------------------

        将自定义函数应用于每个滚动窗口内的最大值。

        参数
        ----------
        function : Callable
            要应用的自定义函数。
        **func_kwds : Any
            要传递给函数的附加关键字参数。

        返回
        -------
        pd.DataFrame
            应用函数结果的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._apply(self._masked_obj, function, **func_kwds)
        x = x.reindex(self._obj.index)
        x = x[self._min_periods_mask >= self.min_periods]
        return x

    @property
    def values(self) -> pd.DataFrame:
        """
        ===========================================================================

        Returns the raw maximum values within each rolling window.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the maximum values for each window.

        ---------------------------------------------------------------------------

        返回每个滚动窗口内的原始最大值。

        返回
        -------
        pd.DataFrame
            包含每个窗口最大值的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rolling_obj(self._masked_obj, False, None)
        return x

class _min(_max, _meta):
    """
    ===========================================================================

    A class for calculating rolling minimums.

    ---------------------------------------------------------------------------

    用于计算滚动最小值的类。

    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        df_obj: pd.DataFrame,
        window: int,
        min_periods: int
    ):
        """
        ===========================================================================

        Initializes the _min class, setting up for rolling minimum calculations.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input DataFrame.
        window : int
            The size of the rolling window.
        min_periods : int
            Minimum number of observations in window required to have a value.

        ---------------------------------------------------------------------------

        初始化_min类，为滚动最小值计算做准备。

        参数
        ----------
        df_obj : pd.DataFrame
            输入DataFrame。
        window : int
            滚动窗口的大小。
        min_periods : int
            窗口中需要有值的最小观测数。

        ---------------------------------------------------------------------------
        """
        _meta.__init__(self, df_obj, window, min_periods, ts_sort_unit, True)

class _rank(_meta):
    """
    ===========================================================================

    A class for calculating rolling ranks.

    ---------------------------------------------------------------------------

    用于计算滚动排名的类。

    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        df_obj: pd.DataFrame,
        window: int,
        min_periods: int
    ):
        """
        ===========================================================================

        Initializes the _rank class, setting up for rolling rank calculations.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input DataFrame.
        window : int
            The size of the rolling window.
        min_periods : int
            Minimum number of observations in window required to have a value.

        ---------------------------------------------------------------------------

        初始化_rank类，为滚动排名计算做准备。

        参数
        ----------
        df_obj : pd.DataFrame
            输入DataFrame。
        window : int
            滚动窗口的大小。
        min_periods : int
            窗口中需要有值的最小观测数。

        ---------------------------------------------------------------------------
        """
        super().__init__(df_obj, window, min_periods, ts_rank_unit, False)

    def __call__(
        self,
        pct: bool
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Calculates the rolling rank of the DataFrame values.

        Parameters
        ----------
        pct : bool
            Whether to return the rank as a percentage.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the rolling ranks.

        ---------------------------------------------------------------------------

        计算DataFrame值的滚动排名。

        参数
        ----------
        pct : bool
            是否以百分比形式返回排名。

        返回
        -------
        pd.DataFrame
            包含滚动排名的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rolling_obj(self._obj.values, pct, None)
        x = x.reindex(self._obj.index)
        x = x[self._min_periods_mask >= self.min_periods]
        return x

class _rolls():
    """
    ===========================================================================

    A container class for various rolling window operations.

    ---------------------------------------------------------------------------

    各种滚动窗口操作的容器类。

    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        pandas_obj: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None
    ):
        """
        ===========================================================================

        Initializes the _rolls class with the DataFrame, window size, and minimum periods.

        Parameters
        ----------
        pandas_obj : pd.DataFrame
            The input DataFrame for rolling operations.
        window : int
            The size of the rolling window.
        min_periods : Optional[int], optional
            Minimum number of observations in window required to have a value. Defaults to None.

        ---------------------------------------------------------------------------

        使用DataFrame、窗口大小和最小周期初始化_rolls类。

        参数
        ----------
        pandas_obj : pd.DataFrame
            用于滚动操作的输入DataFrame。
        window : int
            滚动窗口的大小。
        min_periods : Optional[int], optional
            窗口中需要有值的最小观测数。默认为 None。

        ---------------------------------------------------------------------------
        """
        self._obj = pandas_obj
        self.window = window
        self.min_periods = min_periods if min_periods is not None else window
        self._max_class = _max(pandas_obj, self.window, self.min_periods)
        self._min_class = _min(pandas_obj, self.window, self.min_periods)
        self._rank_class = _rank(pandas_obj, self.window, self.min_periods)

    def max(
        self,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Returns a rolling maximum object, optionally specifying the number of maximums to retrieve.

        Parameters
        ----------
        count : Optional[int], optional
            The number of maximum values to retrieve from each window. Defaults to None.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the rolling maximums.

        ---------------------------------------------------------------------------

        返回一个滚动最大值对象，可选地指定要检索的最大值数量。

        参数
        ----------
        count : Optional[int], optional
            从每个窗口中检索的最大值数量。默认为 None。

        返回
        -------
        pd.DataFrame
            表示滚动最大值的DataFrame。

        ---------------------------------------------------------------------------
        """
        count = self.window if count is None else count
        x = self._max_class(count)
        return x

    def min(
        self,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Returns a rolling minimum object, optionally specifying the number of minimums to retrieve.

        Parameters
        ----------
        count : Optional[int], optional
            The number of minimum values to retrieve from each window. Defaults to None.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the rolling minimums.

        ---------------------------------------------------------------------------

        返回一个滚动最小值对象，可选地指定要检索的最小值数量。

        参数
        ----------
        count : Optional[int], optional
            从每个窗口中检索的最小值数量。默认为 None。

        返回
        -------
        pd.DataFrame
            表示滚动最小值的DataFrame。

        ---------------------------------------------------------------------------
        """
        count = self.window if count is None else count
        x = self._min_class(count)
        return x

    def ts_rank(
        self,
        pct: bool = True
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Calculates the time-series rank within the rolling window.

        Parameters
        ----------
        pct : bool, optional
            Whether to return the rank as a percentage. Defaults to True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the time-series ranks.

        ---------------------------------------------------------------------------

        计算滚动窗口内的时间序列排名。

        参数
        ----------
        pct : bool, optional
            是否以百分比形式返回排名。默认为 True。

        返回
        -------
        pd.DataFrame
            包含时间序列排名的DataFrame。

        ---------------------------------------------------------------------------
        """
        x = self._rank_class(pct)
        return x

