# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union

from ..tools.core import fillna as fillna_func

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

    Groups and ranks a DataFrame based on specified rules.

    This function groups the DataFrame by its index and applies ranking and
    binning rules to its columns.

    ---------------------------------------------------------------------------

    根据指定规则对 DataFrame 进行分组和排名。

    此函数按索引对 DataFrame 进行分组，并对其列应用排名和分箱规则。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be grouped.
    rule : Union[Dict, List]
        A dictionary of rules for specific columns or a list for all.
    pct : bool, optional
        Whether to use percentage-based ranking, by default True.
    order : bool, optional
        If True, grouping is applied sequentially, by default False.
    nlevels : Optional[List[Union[int, str]]], optional
        Column levels to exclude from stacking, by default None.

    ---------------------------------------------------------------------------

    参数
    ----------
    df : pd.DataFrame
        要分组的 DataFrame。
    rule : Union[Dict, List]
        特定列的规则字典或适用于所有列的列表。
    pct : bool, optional
        是否使用基于百分比的排名，默认为 True。
    order : bool, optional
        如果为 True，则顺序应用分组，默认为 False。
    nlevels : Optional[List[Union[int, str]]], optional
        要从堆叠中排除的列级别，默认为 None。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        The grouped and binned DataFrame.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        分组并分箱后的 DataFrame。

    ---------------------------------------------------------------------------
    """
    if isinstance(rule, dict):
        df.index.names = [i if i is not None else 'level_i' + str(j) for j,i in enumerate(df.index.names)]
        df.columns.names = [i if i is not None else 'level_c' + str(j) for j,i in enumerate(df.columns.names)]
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

    Applies weights to a DataFrame.

    This function multiplies a DataFrame by a weight DataFrame. It can handle
    missing values and normalize weights.

    ---------------------------------------------------------------------------

    将权重应用于 DataFrame。

    此函数将一个 DataFrame 乘以一个权重 DataFrame。它可以处理缺失值并归一化权重。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be weighted.
    w_df : Optional[pd.DataFrame], optional
        The DataFrame of weights, by default None.
    fillna : bool, optional
        Whether to forward-fill weights, by default True.
    pct : bool, optional
        If True, normalizes weights to sum to 1, by default True.

    ---------------------------------------------------------------------------

    参数
    ----------
    df : pd.DataFrame
        要加权的 DataFrame。
    w_df : Optional[pd.DataFrame], optional
        权重 DataFrame，默认为 None。
    fillna : bool, optional
        是否前向填充权重，默认为 True。
    pct : bool, optional
        如果为 True，则将权重归一化为总和为 1，默认为 True。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        The weighted DataFrame.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        加权后的 DataFrame。

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
):
    returns = returns.sort_index().rolling(roll).mean() if roll > 1 else returns.sort_index()
    df_obj = (fillna_func(df_obj.sort_index(), returns.index) if fillna else df_obj).shift(shift)
    if weight is not None:
        weight = (fillna_func(weight, returns.index) if fillna else weight).reindex_like(returns)[returns.notnull()]

    df = {i:j for i,j in {'portfolio':df_obj, '1':returns, '2':weight}.items() if j is not None}
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
):
    """
    ===========================================================================

    Selects a slice of a DataFrame based on rank with hysteresis.

    This function selects columns that fall within a specific rank range
    and uses a hysteresis mechanism to reduce turnover.

    ---------------------------------------------------------------------------

    基于带有迟滞效应的排名选择 DataFrame 的切片。

    此函数选择排名在特定范围内的列，并使用迟滞机制来减少换手率。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The DataFrame to be cut.
    left : Union[int, float]
        The left boundary of the rank selection.
    right : Union[int, float]
        The right boundary of the rank selection.
    rng_left : Union[int, float]
        The left hysteresis range.
    rng_right : Union[int, float]
        The right hysteresis range.
    pct : bool, optional
        Whether the ranks are percentage-based, by default True.
    ascending : bool, optional
        The sort order for ranking, by default False.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        要切片的 DataFrame。
    left : Union[int, float]
        排名选择的左边界。
    right : Union[int, float]
        排名选择的右边界。
    rng_left : Union[int, float]
        左侧的迟滞范围。
    rng_right : Union[int, float]
        右侧的迟滞范围。
    pct : bool, optional
        排名是否基于百分比，默认为 True。
    ascending : bool, optional
        排名的排序顺序，默认为 False。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        A boolean DataFrame indicating the selection.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        一个布尔值的 DataFrame，指示所选内容。

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

def part_cut(df_obj: pd.DataFrame,
    left: Union[int, float],
    right: Union[int, float],
    rng_left: Union[int, float],
    rng_right: Union[int, float],
    pct: bool = True,
    ascending: bool = False,
    part = 5
):
    ranger = range(0, df_obj.shape[0], part)
    rangers = {i:np.array(ranger) + i for i in range(part)}
    rangers = {i:j[j < df_obj.shape[0]] for i,j in rangers.items()}
    right_adj = right // part
    df_obj = df_obj.copy()
    obj = None
    for i,j in rangers.items():
        x = cut(df_obj.iloc[j], left, right_adj, rng_left, rng_right, pct, ascending)
        x = x.reindex(df_obj.index).ffill(limit=part-1)
        obj = x if obj is None else (obj.shift(fill_value=False) | x)
        df_obj = df_obj[~obj]
    return obj
