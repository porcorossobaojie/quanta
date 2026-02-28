# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:19:39 2026

@author: Porco Rosso

"""

import numpy as np
from typing import Optional, Callable, Any

__all__ = ['ts_argsort_unit', 'ts_rank_unit', 'ts_sort_unit']

def ts_rank_unit(
    array_obj: np.ndarray,
    cut: Any,
    pct: bool,
    func: Optional[Callable],
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================

    Calculates the time-series rank of the last element in each column.

    This function computes the rank of the last element within a time-series
    array, optionally as a percentage.

    ---------------------------------------------------------------------------

    计算每列中最后一个元素的时间序列排名。

    此函数计算时间序列数组中最后一个元素的排名，可选择以百分比形式表示。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : Any
        Placeholder parameter (not used in current implementation).
    pct : bool
        If True, return rank as a percentage.
    func : Optional[Callable]
        Placeholder parameter (not used in current implementation).
    **kwargs : Any
        Additional keyword arguments (not used in current implementation).

    ---------------------------------------------------------------------------

    参数
    ----------
    array_obj : np.ndarray
        输入的时间序列数组。
    cut : Any
        占位符参数（当前实现中未使用）。
    pct : bool
        如果为 True，则以百分比形式返回排名。
    func : Optional[Callable]
        占位符参数（当前实现中未使用）。
    **kwargs : Any
        附加关键字参数（当前实现中未使用）。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        An array of ranks.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        排名数组。

    ---------------------------------------------------------------------------
    """
    x = (array_obj <= array_obj[-1]).sum(axis=0)
    nans = (~np.isnan(array_obj)).sum(axis=0)
    x = np.where(nans == 0, np.nan, x)
    if pct:
        x = x / nans
    return x


def ts_sort_unit(
    array_obj: np.ndarray,
    cut: int,
    pct: bool,
    func: Optional[Callable],
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================

    Sorts each column of a time-series array and returns a sliced portion.

    This function sorts the input array along axis 0 and returns the top or
    bottom 'cut' elements, optionally applying a function.

    ---------------------------------------------------------------------------

    对时间序列数组的每列进行排序并返回切片部分。

    此函数沿轴 0 对输入数组进行排序，并返回顶部或底部的 'cut' 个元素，
    可选择应用一个函数。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : int
        The number of elements to return. Positive for top, negative for bottom.
    pct : bool
        Placeholder parameter (not used in current implementation).
    func : Optional[Callable]
        An optional function to apply to the sorted and sliced array.
    **kwargs : Any
        Additional keyword arguments for the applied function.

    ---------------------------------------------------------------------------

    参数
    ----------
    array_obj : np.ndarray
        输入的时间序列数组。
    cut : int
        要返回的元素数量。正数表示顶部，负数表示底部。
    pct : bool
        占位符参数（当前实现中未使用）。
    func : Optional[Callable]
        应用于排序和切片数组的可选函数。
    **kwargs : Any
        应用于函数的附加关键字参数。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        The sorted and sliced array.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        排序和切片后的数组。

    ---------------------------------------------------------------------------
    """
    endwith = True if cut > 0 else False
    x = np.ma.sort(array_obj, axis=0, endwith=endwith)
    x = x[:cut] if endwith else x[cut:]
    x = func(x, axis=0, **kwargs) if func is not None else x
    return x


def ts_argsort_unit(
    array_obj: np.ndarray,
    cut: int,
    pct: bool,
    func: Optional[Callable],
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================

    Calculates the time-series argsort of each column and returns a sliced portion.

    This function computes the argsort of the input array along axis 0 and
    returns the indices of the top or bottom 'cut' elements, optionally
    applying a function.

    ---------------------------------------------------------------------------

    计算每列的时间序列 argsort 并返回切片部分。

    此函数沿轴 0 计算输入数组的 argsort，并返回顶部或底部的 'cut' 个元素的索引，
    可选择应用一个函数。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : int
        The number of elements to return. Positive for top, negative for bottom.
    pct : bool
        Placeholder parameter (not used in current implementation).
    func : Optional[Callable]
        An optional function to apply to the sorted and sliced array.
    **kwargs : Any
        Additional keyword arguments for the applied function.

    ---------------------------------------------------------------------------

    参数
    ----------
    array_obj : np.ndarray
        输入的时间序列数组。
    cut : int
        要返回的元素数量。正数表示顶部，负数表示底部。
    pct : bool
        占位符参数（当前实现中未使用）。
    func : Optional[Callable]
        应用于排序和切片数组的可选函数。
    **kwargs : Any
        应用于函数的附加关键字参数。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        The argsorted and sliced array.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        argsort 和切片后的数组。

    ---------------------------------------------------------------------------
    """
    endwith = True if cut > 0 else False
    x = np.ma.array(
        np.ma.argsort(array_obj, axis=0, endwith=endwith),
        mask=(np.sort(array_obj.mask, axis=0) if endwith else np.sort(array_obj.mask, axis=0)[::-1])
    )
    x = x[:cut] if endwith else x[cut:]
    x = func(x, axis=0, **kwargs) if func is not None else x
    return x

