# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:52:37 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, List, Any
from numpy.lib.stride_tricks import as_strided

from src.quanta.libs.utils import flatten_list

__all__ = ['fillna', 'shift', 'log', 'half_life', 'array_roll']

def fillna(
    df_obj: pd.DataFrame,
    fill_list: List[Any]
):
    """
    ===========================================================================

    Forward fills a DataFrame based on a new index.

    This function extends a DataFrame to a new index, forward-filling existing
    values to the new index points.

    ---------------------------------------------------------------------------

    根据新索引前向填充 DataFrame。

    此函数将 DataFrame 扩展到新索引，并将现有值前向填充到新索引点。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The source DataFrame to be filled.
    fill_list : List[Any]
        A list of new index labels to be included.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        需要填充的源 DataFrame。
    fill_list : List[Any]
        需要包含的新索引标签列表。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the combined and forward-filled index.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        一个包含合并和前向填充索引的新 DataFrame。

    ---------------------------------------------------------------------------
    """
    df_obj = df_obj.sort_index()
    old_idx = df_obj.index.to_list()
    index = sorted(fill_list)
    if index[-1] >= old_idx[0]:
        values = df_obj.values
        lst = []
        new_idx = sorted(set(df_obj.index) | set(index))
        position = [new_idx.index(i) for i in old_idx]
        position.append(len(new_idx))
        for i, j in enumerate(position[:-1]):
            repeat = position[i+1] - j
            array = values[i]
            array = array.repeat(repeat)
            lst.append(array.reshape(df_obj.shape[1], -1).T if repeat != 1 else array.reshape(1, -1))
        lst = np.concatenate(lst)
        lst = pd.DataFrame(lst, columns=df_obj.columns, index=new_idx[position[0]:]).reindex(index)
    else:
        lst = pd.DataFrame(np.nan, index=index, columns=df_obj.columns)

    lst.index.name = getattr(fill_list, 'name', df_obj.index.name)
    return lst

def shift(
    df_obj: pd.DataFrame,
    n: int
):
    """
    ===========================================================================

    Conditionally shifts columns with NaN values in the last row.

    This function iteratively shifts down columns that have a NaN value in
    their last row, up to a maximum of 'n' times.

    ---------------------------------------------------------------------------

    有条件地移动最后一行存在 NaN 值的列。

    此函数迭代地向下移动在其最后一行具有 NaN 值的列，最多移动 'n' 次。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The DataFrame to process.
    n : int
        The maximum number of shifts to perform.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        要处理的 DataFrame。
    n : int
        要执行的最大移动次数。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with columns shifted.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        经过列移动处理后的 DataFrame。

    ---------------------------------------------------------------------------
    """
    bools = df_obj.iloc[-1].isnull()

    while n > 0 and bools.any():
        n -= 1
        df_obj.loc[:, bools] = df_obj.loc[:, bools].shift()
        bools = df_obj.iloc[-1].isnull()

    return df_obj

def log(
    df_obj: pd.DataFrame,
    bias_adj: Union[int, float] = 1,
    abs_adj: bool = True
):
    """
    ===========================================================================

    Applies a sign-adjusted logarithmic transformation to a DataFrame.

    This function computes the logarithm of DataFrame values, with an option
    to adjust for the sign of the original values.

    ---------------------------------------------------------------------------

    对 DataFrame 应用符号调整的对数变换。

    此函数计算 DataFrame 值的对数，并提供一个选项来调整原始值的符号。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame.
    bias_adj : Union[int, float], optional
        A bias value to add before the logarithm, by default 1.
    abs_adj : bool, optional
        If True, applies the log to the absolute value and restores the sign.
        By default True.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        输入的 DataFrame。
    bias_adj : Union[int, float], optional
        在对数运算前添加的偏置值，默认为 1。
    abs_adj : bool, optional
        如果为 True，则对绝对值应用对数并恢复符号。默认为 True。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        转换后的 DataFrame。

    ---------------------------------------------------------------------------
    """
    if abs_adj:
        sign = np.sign(df_obj).replace(0, 1)
        x = sign * np.log((df_obj + sign * bias_adj).abs())
    else:
        x = np.log(bias_adj + df_obj)

    return x

def half_life(
    window,
    half_life
):
    L, Lambda = 0.5**(1/half_life), 0.5**(1/half_life)
    W = []
    for i in range(window):
        W.append(Lambda)
        Lambda *= L
    W = np.array(W[::-1])
    return W

def array_roll(
    array_2D: np.ndarray,
    periods: int
) -> np.ndarray:
    """
    ===========================================================================

    Creates rolling windows from a 2D NumPy array.

    Parameters
    ----------
    array_2D : np.ndarray
        The input 2D NumPy array.
    periods : int
        The size of the rolling window.

    Returns
    -------
    np.ndarray
        A 3D NumPy array representing the rolling windows.

    ---------------------------------------------------------------------------

    从2D NumPy数组创建滚动窗口。

    参数
    ----------
    array_2D : np.ndarray
        输入的2D NumPy数组。
    periods : int
        滚动窗口的大小。

    返回
    -------
    np.ndarray
        表示滚动窗口的3D NumPy数组。

    ---------------------------------------------------------------------------
    """
    axis=0

    new_shape = list(array_2D.shape)
    new_shape[axis] = [periods, new_shape[axis] - periods + 1, ]
    new_shape = tuple(flatten_list(new_shape))

    new_strides = list(array_2D.strides)
    new_strides[axis] = [array_2D.strides[axis], array_2D.strides[axis]]
    new_strides = tuple(flatten_list(new_strides))

    window = as_strided(array_2D, shape=new_shape, strides=new_strides)
    window = window.transpose(1, 0, 2)
    return window
