# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:58:12 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from itertools import product
from typing import Optional, Dict, List, Union

from quanta.libs._pandas.tools.core import fillna as fillna_func

__all__ = ['group', 'weight', 'portfolio', 'cut', 'roll_weight', 'd_cut']

@njit(parallel=True, cache=True, nopython=True)
def fast_rank(data_2d, rule):
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
    is_multi = bool(df.columns.nlevels - 1)
    rule = {i:np.array(j) for i,j in rule.items()} if isinstance(rule, dict) else np.array(rule)

    if not is_multi:
        df = pd.DataFrame(fast_rank(df.values, rule), index=df.index, columns=df.columns)
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

def group_old(
    df: pd.DataFrame,
    rule: Union[Dict, List],
    pct: bool = True,
    order: bool = False,
    nlevels: Optional[List[Union[int, str]]] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Groups and ranks a DataFrame based on specified rules, typically for
    factor grouping and binning.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be grouped.
    rule : Union[Dict, List]
        A dictionary of rules for specific columns or a list for all columns.
    pct : bool
        Whether to use percentage-based ranking. Default is True.
    order : bool
        If True, grouping is applied sequentially based on previously binned
        columns. Default is False.
    nlevels : Optional[List[Union[int, str]]]
        Column levels to be preserved during the stacking process.

    Returns
    -------
    pd.DataFrame
        The grouped and binned DataFrame.
    ---------------------------------------------------------------------------
    根据指定规则对 DataFrame 进行分组和排名, 通常用于因子分组和分箱.

    参数
    ----
    df : pd.DataFrame
        要分组的 DataFrame.
    rule : Union[Dict, List]
        特定列的规则字典或适用于所有列的列表.
    pct : bool
        是否使用基于百分比的排名. 默认为 True.
    order : bool
        如果为 True, 则基于之前已分箱的列顺序应用分组. 默认为 False.
    nlevels : Optional[List[Union[int, str]]]
        在堆叠过程中要保留的列层级.

    返回
    ----
    pd.DataFrame
        分组并分箱后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    if isinstance(rule, dict):
        df.index.names = [i if i is not None else 'level_i' + str(j) for j, i in enumerate(df.index.names)]
        df.columns.names = [i if i is not None else 'level_c' + str(j) for j, i in enumerate(df.columns.names)]
        ind_keys = list(df.index.names)
        col_nlevels = [0] if nlevels is None else nlevels
        col_nlevels = [i if isinstance(i, int) else df.columns.name.index(i) for i in col_nlevels]
        df = df.stack(sorted(set(range(df.columns.nlevels)) - set(col_nlevels)))
        df = df.loc[:, list(rule.keys())]
        used_keys = []
        for k, i in enumerate(df.columns):
            df[i] = df.groupby(ind_keys + used_keys)[i].rank(pct=pct)
            df[i] = pd.cut(df[i], rule[i], labels=[str([rule[i][j], rule[i][j+1]]) for j in range(len(rule[i]) - 1)])
            if order:
                used_keys.append(i)
        df = df.unstack(list(range(df.index.nlevels)[-1 * len(col_nlevels):]))
    else:
        df = df.rank(axis=1, pct=pct).round(9)
        col_nlevels = df.columns.nlevels
        df = df.stack(list(range(col_nlevels)))
        df = pd.cut(df, rule, labels=[str([rule[i], rule[i+1]]) for i in range(len(rule) - 1)])
        df = df.unstack(list(range(df.index.nlevels)[-1 * col_nlevels:]))
    return df


def weight(
    df: pd.DataFrame,
    w_df: Optional[pd.DataFrame] = None,
    fillna: bool = True,
    pct: bool = True,
) -> pd.DataFrame:
    """
    ===========================================================================
    Applies weights to a DataFrame, supporting forward-filling and
    normalization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be weighted.
    w_df : Optional[pd.DataFrame]
        The DataFrame of weights. Default is None.
    fillna : bool
        Whether to forward-fill weights to match the index of df.
        Default is True.
    pct : bool
        If True, normalizes weights to sum to 1 across columns.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The weighted DataFrame.
    ---------------------------------------------------------------------------
    将权重应用于 DataFrame, 支持前向填充和归一化.

    参数
    ----
    df : pd.DataFrame
        要加权的 DataFrame.
    w_df : Optional[pd.DataFrame]
        权重 DataFrame. 默认为 None.
    fillna : bool
        是否前向填充权重以匹配 df 的索引. 默认为 True.
    pct : bool
        如果为 True, 则将权重归一化为行总和为 1. 默认为 True.

    返回
    ----
    pd.DataFrame
        加权后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    if w_df is not None:
        if fillna:
            w_df = fillna_func(w_df, df.index)
        w_df = w_df.reindex_like(df)
        w_df[df.isnull()] = pd.NA
        if pct:
            w_df = w_df.div(w_df.sum(axis=1), axis=0)
        return df * w_df
    else:
        if pct:
            return df.div(df.notnull().sum(axis=1), axis=0)
        else:
            return df


def portfolio(
    df_obj: pd.DataFrame,
    returns: pd.DataFrame,
    weight: Optional[pd.DataFrame] = None,
    shift: int = 1,
    roll: int = 1,
    fillna: bool = False
) -> pd.DataFrame:
    """
    ===========================================================================
    Calculates group returns (portfolio returns) based on group assignments
    and asset returns.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The DataFrame containing group labels (e.g., output of group()).
    returns : pd.DataFrame
        The DataFrame of asset returns.
    weight : Optional[pd.DataFrame]
        The weights of assets. Default is None.
    shift : int
        The number of periods to shift group assignments forward.
        Default is 1.
    roll : int
        The rolling window for asset returns. Default is 1.
    fillna : bool
        Whether to forward-fill group assignments and weights.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing average or weighted returns for each group.
    ---------------------------------------------------------------------------
    根据分组分配和资产收益率计算组收益 (组合收益).

    参数
    ----
    df_obj : pd.DataFrame
        包含分组标签的 DataFrame (例如 group() 的输出).
    returns : pd.DataFrame
        资产收益率的 DataFrame.
    weight : Optional[pd.DataFrame]
        资产的权重. 默认为 None.
    shift : int
        将分组分配前移的周期数. 默认为 1.
    roll : int
        资产收益率的滚动窗口大小. 默认为 1.
    fillna : bool
        是否前向填充分组分配和权重. 默认为 False.

    返回
    ----
    pd.DataFrame
        包含每个组的平均或加权收益率的 DataFrame.
    ---------------------------------------------------------------------------
    """
    returns = returns.sort_index().rolling(roll).mean().shift(-1 * (roll-1)) if roll > 1 else returns.sort_index()
    df_obj = (fillna_func(df_obj.sort_index(), returns.index) if fillna else df_obj).shift(shift)
    if weight is not None:
        weight = (fillna_func(weight, returns.index) if fillna else weight).reindex_like(returns)[returns.notnull()]

    df = {i: j for i, j in {'portfolio': df_obj, '1': returns, '2': weight}.items() if j is not None}
    df = pd.concat(df, axis=1).stack().set_index('portfolio', append=True)
    if weight is not None:
        df['1'] = df['1'] * df['2']
        group_obj = df.groupby([df.index.names[0], df.index.names[2]])
        df = group_obj['1'].sum() / group_obj['2'].sum()
    else:
        df = df['1'].groupby([df.index.names[0], df.index.names[2]]).mean()
    df = df.unstack().astype('float64').sort_index(axis=0).sort_index(axis=1)
    return df


def cut(
    df_obj: pd.DataFrame,
    left: Union[int, float],
    right: Union[int, float],
    rng_left: Union[int, float],
    rng_right: Union[int, float],
    pct: bool = True,
    ascending: bool = False
) -> pd.DataFrame:
    """
    ===========================================================================
    Selects a slice of a DataFrame based on rank with a hysteresis
    mechanism to reduce turnover.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame (e.g., factor values).
    left : Union[int, float]
        The lower bound of the target rank range.
    right : Union[int, float]
        The upper bound of the target rank range.
    rng_left : Union[int, float]
        The buffer range on the left for hysteresis.
    rng_right : Union[int, float]
        The buffer range on the right for hysteresis.
    pct : bool
        Whether ranks are calculated as percentages. Default is True.
    ascending : bool
        The sort order for ranking. Default is False.

    Returns
    -------
    pd.DataFrame
        A boolean DataFrame indicating selected assets.
    ---------------------------------------------------------------------------
    基于带有迟滞机制的排名选择 DataFrame 的切片, 以减少换手率.

    参数
    ----
    df_obj : pd.DataFrame
        输入 DataFrame (例如因子值).
    left : Union[int, float]
        目标排名范围的下界.
    right : Union[int, float]
        目标排名范围的上界.
    rng_left : Union[int, float]
        左侧迟滞缓冲范围.
    rng_right : Union[int, float]
        右侧迟滞缓冲范围.
    pct : bool
        排名是否以百分比计算. 默认为 True.
    ascending : bool
        排名的排序顺序. 默认为 False.

    返回
    ----
    pd.DataFrame
        指示所选资产的布尔值 DataFrame.
    ---------------------------------------------------------------------------
    """
    role = right - left
    lst = []
    rank = df_obj.rank(axis=1, pct=pct, ascending=ascending)
    j = rank.iloc[0]
    j = (j >= left) & (j <= right)
    lst.append(j.values)
    for i, j in rank.iloc[1:].iterrows():
        hold = (j >= left - rng_left) & (j <= right + rng_right) & lst[-1]
        lens = int(role * j.notnull().sum()) if pct else role
        updates = lens - hold.sum()
        if updates > 0:
            j = j[(~hold) & (j >= left)].sort_values().head(updates)
            hold[j.index] = True
        elif updates < 0:
            hold[~hold.index.isin(j[hold].sort_values().head(lens).index)] = False
        lst.append(hold.values)
    lst = pd.DataFrame(np.vstack(lst), index=df_obj.index, columns=df_obj.columns)
    return lst

def d_cut(df_obj, count, max_count, delay):
    val = df_obj.values.copy()
    result = np.zeros_like(val)
    mask = ~np.isnan(val[0])
    masked_val = val[0][mask]
    ranks = (-masked_val).argsort().argsort() + 1
    result[0, mask] = np.where(ranks <= count if isinstance(count, int) else count[0], ranks, 0)
    for i in range(1, val.shape[0]):
        arr = val[i]
        hold = result[i-1]
        must_hold = (result[:i][-delay:] > 0).sum(axis=0)
        must_hold = (must_hold >=1) & (must_hold < delay)
        mask = ~np.isnan(arr)
        rank = (-arr[mask]).argsort().argsort() + 1        
        result[i, mask] = np.where(
            (
                must_hold[mask] |
                (rank <= (count if isinstance(count, int) else count[i]) + (max_count if isinstance(max_count, int) else max_count[i]))
            ),
            hold[mask],
            0
        )
        change_count = (count if isinstance(count, int) else count[i]) - (result[i] > 0).sum()
        if change_count > 0:
            mask2 = (~np.isnan(arr)) & (result[i] <= 0)
            rank2 = (-arr[mask2]).argsort().argsort() + 1        
            result[i, mask2] = np.where(
                (rank2 <= change_count),
                rank2,
                result[i, mask2]
            )
        result[i, mask] = np.where(result[i, mask] > 0, rank, 0)
    result = pd.DataFrame(result, index=df_obj.index, columns=df_obj.columns)        
    return result    

def roll_weight(
    df_obj: pd.DataFrame,
    weight_array: Union[List, np.ndarray, pd.Series],
    fillna: bool = 0
) -> pd.DataFrame:
    """
    ===========================================================================
    Calculates a rolling weighted average of a DataFrame using a specified
    weight array.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame.
    weight_array : Union[List, np.ndarray, pd.Series]
        The array of weights to be applied to the rolling window.
    fix_na : bool
        Whether to adjust weights to account for missing values in the window.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The rolling weighted average DataFrame.
    ---------------------------------------------------------------------------
    使用指定的权重数组计算 DataFrame 的滚动加权平均值.

    参数
    ----
    df_obj : pd.DataFrame
        输入 DataFrame.
    weight_array : Union[List, np.ndarray, pd.Series]
        要应用于滚动窗口的权重数组.
    fix_na : bool
        是否调整权重以考虑窗口中的缺失值. 默认为 True.

    返回
    ----
    pd.DataFrame
        滚动加权平均后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    window = len(weight_array)
    weight_array = np.array(weight_array)
    w_adj = df_obj.notnull().astype(int)
    x = df_obj.fillna(fillna) if fillna is not None else df_obj
    
    up = np.einsum("ijk, j -> ik", pd.tools.array_roll(x.values, window), weight_array)
    down = np.einsum("ijk, j -> ik", pd.tools.array_roll(w_adj.values, window), weight_array)
    x = pd.DataFrame(up / down, index=df_obj.index[window-1:], columns=df_obj.columns)
    x = x.reindex_like(df_obj)
    return x
    
    
