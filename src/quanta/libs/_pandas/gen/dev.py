# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:49:40 2026

@author: admin
"""
import numpy as np
import pandas as pd
from numba import njit, prange
from itertools import product
from typing import Dict, List, Union

from ..tools.core import fillna as fillna_func


@njit(parallel=True, cache=True, nopython=True)
def fast_rank(
    data_2d: np.ndarray,
    rule: np.ndarray
) -> np.ndarray:
    """
    ===========================================================================
    Numba-compiled vectorized ranking of each row into bins defined by the
    rule array, ignoring NaN values.

    Parameters
    ----------
    data_2d : np.ndarray
        A 2D array of values to rank per row.
    rule : np.ndarray
        Sorted bin edges (percentile thresholds).

    Returns
    -------
    np.ndarray
        Array of bin labels (1-based), NaN preserved.
    ---------------------------------------------------------------------------
    基于 Numba 编译的逐行向量化排名, 按 rule 数组定义的区间分箱, 忽略 NaN 值.

    参数
    ----
    data_2d : np.ndarray
        待逐行排名的二维数组.
    rule : np.ndarray
        排序后的分箱边界 (百分位阈值).

    返回
    ----
    np.ndarray
        分箱标签数组 (从 1 开始), 保留 NaN.
    ---------------------------------------------------------------------------
    """
    result = np.full(data_2d.shape, np.nan)
    for i in prange(data_2d.shape[0]):
        mask = ~np.isnan(data_2d[i])
        slice_data = data_2d[i][mask]
        if len(slice_data):
            count = slice_data.shape[0]
            slice_data = slice_data.argsort().argsort() + 1
            slice_data = slice_data / count
            slice_data = np.searchsorted(rule, slice_data, side='left')
            slice_data = np.fmax(1, np.fmin(len(rule) - 1, slice_data))
            result[i, mask] = slice_data
    return result


def group(
    df: pd.DataFrame,
    rule: Union[Dict, List],
    order: bool = True,
) -> pd.DataFrame:
    """
    ===========================================================================
    Groups and ranks a DataFrame based on specified rules, typically for
    factor grouping and binning. Supports sequential ordering when multiple
    keys are provided.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be grouped.
    rule : Union[Dict, List]
        A dictionary of rules for specific columns or a list for all columns.
    order : bool
        If True, grouping is applied sequentially based on previously binned
        columns. Default is True.

    Returns
    -------
    pd.DataFrame
        The grouped and binned DataFrame.
    ---------------------------------------------------------------------------
    根据指定规则对 DataFrame 进行分组和排名, 通常用于因子分组和分箱. 当提供多个
    键时支持顺序分组.

    参数
    ----
    df : pd.DataFrame
        要分组的 DataFrame.
    rule : Union[Dict, List]
        特定列的规则字典或适用于所有列的列表.
    order : bool
        如果为 True, 则基于之前已分箱的列顺序应用分组. 默认为 True.

    返回
    ----
    pd.DataFrame
        分组并分箱后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    is_multi = bool(df.columns.nlevels - 1)
    rule = {i:np.array(j) for i,j in rule.items()} if isinstance(rule, dict) else np.array(rule)

    if not is_multi:
        df = pd.DataFrame(fast_rank(df.values, rule), index=df.index, columns=df.columns).astype('Int64')
    else:
        if isinstance(rule, np.ndarray):
            rule = {i: rule for i in df.columns.get_level_values(0).unique()}
        keys = list(rule.keys())
        x = df.sort_index(axis=1)
        cols = x.columns.get_level_values(-1).value_counts() == len(keys)
        cols = cols[cols].index
        x = x.loc[:, x.columns.get_level_values(-1).isin(cols)]
        arrays = np.full((x.index.shape[0], cols.shape[0], len(keys)), np.nan)
        if order:
            arrays[:, :, 0] = fast_rank(x[keys[0]].values, rule[keys[0]])
            ruled = [range(1, len(rule[keys[0]]))]
            for i in range(1, len(keys)):
                flat_values = arrays[:, :, :i]
                unique_keys = np.array(list(product(*ruled)))
                result = (flat_values[:, :, np.newaxis, :] == unique_keys[np.newaxis, np.newaxis, :, :])
                result = result.all(axis=-1)
                result = np.where(result, x[keys[i]].values[:, :, np.newaxis], np.nan)
                result = result.transpose(2, 0, 1).reshape(-1, result.shape[1])
                result = fast_rank(result, rule[keys[i]])
                result = result.reshape(len(unique_keys), -1, result.shape[-1])
                result = np.nansum(result, axis=0)
                arrays[:, :, i] = result
                ruled.append(range(1, len(rule[keys[i]])))
        else:
            for i in range(len(keys)):
                arrays[:, :, i] =fast_rank(x[keys[i]].values, rule[keys[i]])
        df = pd.DataFrame(arrays.reshape(-1, len(keys))).fillna(-1).astype(int).astype(str)
        df = pd.Series(df.values.tolist()).str.join('_')
        df = pd.DataFrame(df.values.reshape(x.shape[0], -1), index=x.index, columns=cols)
        df = df.stack()
        df = df[~df.str.contains('-1',  regex=False)].unstack()
    return df
