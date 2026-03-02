# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:58:12 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union

from quanta.libs._pandas.tools.core import fillna as fillna_func

__all__ = ['group', 'weight', 'portfolio', 'cut', 'part_cut']


def group(
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
    returns = returns.sort_index().rolling(roll).mean() if roll > 1 else returns.sort_index()
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


def part_cut(
    df_obj: pd.DataFrame,
    left: Union[int, float],
    right: Union[int, float],
    rng_left: Union[int, float],
    rng_right: Union[int, float],
    pct: bool = True,
    ascending: bool = False,
    part: int = 5
) -> pd.DataFrame:
    """
    ===========================================================================
    Performs a phased cut operation by dividing the DataFrame into parts
    to spread turnover.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame.
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
    part : int
        The number of parts to divide the data into. Default is 5.

    Returns
    -------
    pd.DataFrame
        A boolean DataFrame indicating selected assets across all parts.
    ---------------------------------------------------------------------------
    通过将 DataFrame 分成多个部分来执行分阶段切片操作, 以分散换手率.

    参数
    ----
    df_obj : pd.DataFrame
        输入 DataFrame.
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
    part : int
        将数据分成的部分数量. 默认为 5.

    返回
    ----
    pd.DataFrame
        指示所有部分中所选资产的布尔值 DataFrame.
    ---------------------------------------------------------------------------
    """
    ranger = range(0, df_obj.shape[0], part)
    rangers = {i: np.array(ranger) + i for i in range(part)}
    rangers = {i: j[j < df_obj.shape[0]] for i, j in rangers.items()}
    right_adj = right // part
    df_obj = df_obj.copy()
    obj = None
    for i, j in rangers.items():
        x = cut(df_obj.iloc[j], left, right_adj, rng_left, rng_right, pct, ascending)
        x = x.reindex(df_obj.index).ffill(limit=part-1)
        obj = x if obj is None else (obj.shift(fill_value=False) | x)
        df_obj = df_obj[~obj]
    return obj
