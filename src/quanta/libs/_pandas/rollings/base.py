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
    func: Optional[Callable] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================
    Calculates the time-series rank of the last element in each column of a
    NumPy array.

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : Any
        Placeholder parameter for compatibility with other unit functions.
    pct : bool
        If True, return rank as a percentage (rank / non-nan count).
    func : Optional[Callable]
        Placeholder parameter for compatibility. Default is None.
    **kwargs : Any
        Additional keyword arguments. Default is empty.

    Returns
    -------
    np.ndarray
        An array containing the ranks for each column.
    ---------------------------------------------------------------------------
    计算 NumPy 数组每列中最后一个元素的时间序列排名.

    参数
    ----
    array_obj : np.ndarray
        输入的时间序列数组.
    cut : Any
        用于与其他单元函数兼容的占位符参数.
    pct : bool
        如果为 True, 则以百分比形式返回排名 (排名 / 非 NaN 计数).
    func : Optional[Callable]
        用于兼容性的占位符参数. 默认为 None.
    **kwargs : Any
        附加关键字参数. 默认为空.

    返回
    ----
    np.ndarray
        包含每列排名的数组.
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
    func: Optional[Callable] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================
    Sorts each column of a time-series array and returns a sliced portion,
    optionally applying an aggregation function.

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : int
        The number of elements to return. Positive for top, negative for bottom.
    pct : bool
        Placeholder parameter for compatibility.
    func : Optional[Callable]
        An optional function to apply to the sorted and sliced array.
    **kwargs : Any
        Additional keyword arguments for the applied function.

    Returns
    -------
    np.ndarray
        The sorted and sliced array, or the result of the applied function.
    ---------------------------------------------------------------------------
    对时间序列数组的每列进行排序并返回切片部分, 可选择应用聚合函数.

    参数
    ----
    array_obj : np.ndarray
        输入的时间序列数组.
    cut : int
        要返回的元素数量. 正数表示顶部, 负数表示底部.
    pct : bool
        用于兼容性的占位符参数.
    func : Optional[Callable]
        应用于排序和切片数组的可选函数.
    **kwargs : Any
        应用于函数的附加关键字参数.

    返回
    ----
    np.ndarray
        排序和切片后的数组, 或应用函数后的结果.
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
    func: Optional[Callable] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    ===========================================================================
    Calculates the time-series argsort of each column and returns a sliced
    portion, optionally applying an aggregation function.

    Parameters
    ----------
    array_obj : np.ndarray
        The input time-series array.
    cut : int
        The number of elements to return. Positive for top, negative for bottom.
    pct : bool
        Placeholder parameter for compatibility.
    func : Optional[Callable]
        An optional function to apply to the sliced indices.
    **kwargs : Any
        Additional keyword arguments for the applied function.

    Returns
    -------
    np.ndarray
        The argsorted indices, or the result of the applied function.
    ---------------------------------------------------------------------------
    计算每列的时间序列 argsort 并返回切片部分, 可选择应用聚合函数.

    参数
    ----
    array_obj : np.ndarray
        输入的时间序列数组.
    cut : int
        要返回的元素数量. 正数表示顶部, 负数表示底部.
    pct : bool
        用于兼容性的占位符参数.
    func : Optional[Callable]
        应用于切片索引的可选函数.
    **kwargs : Any
        应用于函数的附加关键字参数.

    返回
    ----
    np.ndarray
        argsort 索引, 或应用函数后的结果.
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
