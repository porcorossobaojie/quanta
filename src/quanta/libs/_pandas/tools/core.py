# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:52:37 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, List, Any
from numpy.lib.stride_tricks import as_strided

from quanta.libs.utils import flatten_list

__all__ = ['fillna', 'shift', 'log', 'half_life', 'array_roll']


def fillna(
    df_obj: pd.DataFrame,
    fill_list: List[Any]
) -> pd.DataFrame:
    """
    ===========================================================================
    Forward fills a DataFrame based on a new index, effectively extending it.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The source DataFrame to be filled.
    fill_list : List[Any]
        A list of new index labels to be included in the result.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the combined and forward-filled index.
    ---------------------------------------------------------------------------
    根据新索引前向填充 DataFrame, 有效地扩展它.

    参数
    ----
    df_obj : pd.DataFrame
        要填充的源 DataFrame.
    fill_list : List[Any]
        要包含在结果中的新索引标签列表.

    返回
    ----
    pd.DataFrame
        具有合并和前向填充索引的新 DataFrame.
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
) -> pd.DataFrame:
    """
    ===========================================================================
    Iteratively shifts down columns that have a NaN value in their last row.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The DataFrame to process.
    n : int
        The maximum number of shifts to perform for each column.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with columns shifted as needed.
    ---------------------------------------------------------------------------
    迭代地向下移动在其最后一行具有 NaN 值的列.

    参数
    ----
    df_obj : pd.DataFrame
        要处理的 DataFrame.
    n : int
        每列执行的最大移动次数.

    返回
    ----
    pd.DataFrame
        根据需要进行列移动处理后的 DataFrame.
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
) -> pd.DataFrame:
    """
    ===========================================================================
    Applies a sign-preserved or standard logarithmic transformation.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame.
    bias_adj : Union[int, float]
        A bias value to add before the logarithm. Default is 1.
    abs_adj : bool
        If True, applies the log to the absolute value and restores the sign.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.
    ---------------------------------------------------------------------------
    应用符号保留或标准的对数变换.

    参数
    ----
    df_obj : pd.DataFrame
        输入的 DataFrame.
    bias_adj : Union[int, float]
        在对数运算前添加的偏置值. 默认为 1.
    abs_adj : bool
        如果为 True, 则对绝对值应用对数并恢复符号. 默认为 True.

    返回
    ----
    pd.DataFrame
        转换后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    if abs_adj:
        sign = np.sign(df_obj).replace(0, 1)
        x = sign * np.log((df_obj + sign * bias_adj).abs())
    else:
        x = np.log(bias_adj + df_obj)

    return x


def half_life(
    window: int,
    half_life: Union[int, float]
) -> np.ndarray:
    """
    ===========================================================================
    Generates an exponential decay weight array based on a given half-life.

    Parameters
    ----------
    window : int
        The size of the window (length of the weight array).
    half_life : Union[int, float]
        The decay half-life.

    Returns
    -------
    np.ndarray
        The generated weight array.
    ---------------------------------------------------------------------------
    根据给定的半衰期生成指数衰减权重数组.

    参数
    ----
    window : int
        窗口大小 (权重数组的长度).
    half_life : Union[int, float]
        衰减半衰期.

    返回
    ----
    np.ndarray
        生成的权重数组.
    ---------------------------------------------------------------------------
    """
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
    Creates a rolling window view of a 2D NumPy array using stride tricks.

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
    使用步长技巧从 2D NumPy 数组创建滚动窗口视图.

    参数
    ----
    array_2D : np.ndarray
        输入的 2D NumPy 数组.
    periods : int
        滚动窗口的大小.

    返回
    ----
    np.ndarray
        表示滚动窗口的 3D NumPy 数组.
    ---------------------------------------------------------------------------
    """
    axis = 0

    new_shape = list(array_2D.shape)
    new_shape[axis] = [periods, new_shape[axis] - periods + 1, ]
    new_shape = tuple(flatten_list(new_shape))

    new_strides = list(array_2D.strides)
    new_strides[axis] = [array_2D.strides[axis], array_2D.strides[axis]]
    new_strides = tuple(flatten_list(new_strides))

    window = as_strided(array_2D, shape=new_shape, strides=new_strides)
    window = window.transpose(1, 0, 2)
    return window
