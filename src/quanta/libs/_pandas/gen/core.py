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
from ..tools.core import fillna as fillna_func

__all__ = ['group', 'weight', 'portfolio', 'cut', 'roll_weight', 'd_cut']

@njit(cache=True, nopython=True)
def _bin_table(n_cols: int, rule: np.ndarray) -> np.ndarray:
    """
    ===========================================================================
    Builds a lookup table mapping group size and sorted position to a bin
    label for percentile rules.

    Parameters
    ----------
    n_cols : int
        The maximum group size (number of columns).
    rule : np.ndarray
        Sorted bin edges (percentile thresholds).

    Returns
    -------
    np.ndarray
        Table where table[c, j] is the bin label of the (j+1)-th smallest
        value in a group of size c.
    ---------------------------------------------------------------------------
    构建将组大小和排序位置映射到百分位规则分箱标签的查找表.

    参数
    ----
    n_cols : int
        最大组大小 (列数).
    rule : np.ndarray
        排序后的分箱边界 (百分位阈值).

    返回
    ----
    np.ndarray
        查找表, table[c, j] 为大小为 c 的组中第 (j+1) 小的值所在分箱.
    ---------------------------------------------------------------------------
    """
    n_rule = rule.shape[0]
    table = np.empty((n_cols + 1, n_cols), dtype=np.int64)
    for c in range(1, n_cols + 1):
        for j in range(c):
            pct = (j + 1) / c
            b = np.searchsorted(rule, pct, side='left')
            b = max(min(b, n_rule - 1), 1)
            table[c, j] = b
    return table


@njit(parallel=True, cache=True, nopython=True)
def fast_rank(data_2d: np.ndarray, rule: np.ndarray) -> np.ndarray:
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
    n_rows, n_cols = data_2d.shape
    table = _bin_table(n_cols, rule)
    result = np.full(data_2d.shape, np.nan)
    for i in prange(n_rows):
        row = data_2d[i]
        mask = ~np.isnan(row)
        pos = np.flatnonzero(mask)
        n = pos.shape[0]
        if n:
            order = np.argsort(row[mask])       # order[j] = original col of j-th smallest
            result[i, pos[order]] = table[n, :n]
    return result


@njit(parallel=True, cache=True, nopython=True)
def _fast_rank_grouped(values: np.ndarray, codes: np.ndarray, valid: np.ndarray,
                       rule: np.ndarray) -> np.ndarray:
    """
    ===========================================================================
    Ranks values within groups per row and maps ranks to percentile bins.

    For each row, values are ranked across the column axis inside the groups
    defined by `codes` (only where `valid` and the value is not NaN), then
    mapped to bins. Excluded cells are set to 0.0, replicating the original
    np.nansum(..., 0) behaviour where an all-NaN slice sums to 0 instead of
    NaN. 0 is never a legal bin; downstream logic treats it as "no rank".

    Parameters
    ----------
    values : np.ndarray
        A 2D array of values to rank.
    codes : np.ndarray
        A 2D integer array of group codes per cell.
    valid : np.ndarray
        A 2D boolean array indicating which cells participate.
    rule : np.ndarray
        Sorted bin edges (percentile thresholds).

    Returns
    -------
    np.ndarray
        Array of bin labels (1-based); excluded cells are 0.0.
    ---------------------------------------------------------------------------
    按组对每行的值进行排名并映射到百分位分箱.

    每行中, 值仅在 `codes` 定义的组内 (且 `valid` 为 True, 值非 NaN) 跨列排名,
    然后映射到分箱. 被排除的单元格设为 0.0, 复现原始 np.nansum(..., 0) 的行为,
    即全 NaN 切片求和为 0 而非 NaN. 0 永远不是合法分箱; 下游逻辑将其视为
    "无排名".

    参数
    ----
    values : np.ndarray
        待排名的二维数组.
    codes : np.ndarray
        每个单元格的组编码二维整数数组.
    valid : np.ndarray
        指示哪些单元格参与排名的二维布尔数组.
    rule : np.ndarray
        排序后的分箱边界 (百分位阈值).

    返回
    ----
    np.ndarray
        分箱标签数组 (从 1 开始); 被排除的单元格为 0.0.
    ---------------------------------------------------------------------------
    """
    n_rows, n_cols = values.shape
    table = _bin_table(n_cols, rule)
    result = np.zeros(values.shape)             # 0.0 = "no rank"
    for r in prange(n_rows):
        vbuf = np.empty(n_cols, np.float64)
        cbuf = np.empty(n_cols, np.int64)
        pbuf = np.empty(n_cols, np.int64)
        n = 0
        for c in range(n_cols):
            if valid[r, c] and not np.isnan(values[r, c]):
                vbuf[n] = values[r, c]
                cbuf[n] = codes[r, c]
                pbuf[n] = c
                n += 1
        if n == 0:
            continue
        # sort by (code, value): key = code * n + rank_of_value (no np.lexsort in numba)
        o1 = np.argsort(vbuf[:n])
        rank = np.empty(n, dtype=np.int64)
        rank[o1] = np.arange(n)
        order = np.argsort(cbuf[:n] * np.int64(n) + rank)
        k = 0
        while k < n:
            code = cbuf[order[k]]
            g0 = k
            k += 1
            while k < n and cbuf[order[k]] == code:
                k += 1
            gsize = k - g0
            for j in range(g0, k):
                result[r, pbuf[order[j]]] = table[gsize, j - g0]
    return result


def _join_labels(arrays: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    """
    ===========================================================================
    Joins per-key bin labels into 'b0_b1_...' strings per cell.

    Cells with any NaN key become '-1_...' and are dropped later. Labels are
    packed into integers, deduplicated, formatted only for unique combos, then
    gathered back -- avoiding per-element string work on the full matrix.

    Parameters
    ----------
    arrays : np.ndarray
        A 3D array of bin labels per row, column and key.
    bin_counts : np.ndarray
        The number of bins per key, used for packing.

    Returns
    -------
    np.ndarray
        A 2D array of joined label strings per cell.
    ---------------------------------------------------------------------------
    将每个键的分箱标签连接为每个单元格的 'b0_b1_...' 字符串.

    含任意 NaN 键的单元格会变成 '-1_...', 之后被丢弃. 标签先打包为整数, 去重后
    仅对唯一组合进行格式化, 再还原回矩阵 -- 避免对整个矩阵做逐元素字符串操作.

    参数
    ----
    arrays : np.ndarray
        每个键在行, 列上的分箱标签三维数组.
    bin_counts : np.ndarray
        每个键的分箱数量, 用于打包.

    返回
    ----
    np.ndarray
        每个单元格连接后的标签字符串二维数组.
    ---------------------------------------------------------------------------
    """
    n_rows, n_cols, n_keys = arrays.shape
    strides = np.empty(n_keys, dtype=np.int64)
    s = 1
    for j in range(n_keys):
        strides[j] = s
        s *= bin_counts[j] + 2
    codes = np.zeros((n_rows, n_cols), dtype=np.int64)
    for j in range(n_keys):
        a = arrays[:, :, j]
        codes += (np.where(np.isnan(a), -1.0, a).astype(np.int64) + 1) * strides[j]
    uniq, inv = np.unique(codes, return_inverse=True)
    strings = np.empty(uniq.shape[0], dtype='U64')
    for idx, c in enumerate(uniq.tolist()):
        parts = []
        for j in range(n_keys):
            b = (c // strides[j]) % (bin_counts[j] + 2) - 1
            parts.append(str(b))
        strings[idx] = '_'.join(parts)
    return strings[inv].reshape(n_rows, n_cols)


def group(df: pd.DataFrame, rule: Union[Dict, List], order: bool = True) -> pd.DataFrame:
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
    rule = {k: np.asarray(v) for k, v in rule.items()} if isinstance(rule, dict) else np.asarray(rule)

    if not is_multi:
        return pd.DataFrame(fast_rank(df.values, rule), index=df.index, columns=df.columns)

    if isinstance(rule, np.ndarray):
        rule = {i: rule for i in df.columns.get_level_values(0).unique()}

    keys = list(rule.keys())
    x = df.sort_index(axis=1)
    cnt = x.columns.get_level_values(-1).value_counts()
    cols = cnt[cnt == len(keys)].index
    x = x.loc[:, x.columns.get_level_values(-1).isin(cols)]

    n_rows, n_cols, n_keys = x.shape[0], len(cols), len(keys)

    # contiguous value blocks per key, all in the same (cols) column order
    values_stack = np.empty((n_keys, n_rows, n_cols), dtype=np.float64)
    for i, k in enumerate(keys):
        values_stack[i] = np.ascontiguousarray(x[k].values)

    # strides to pack the bins of keys[0..i-1] into a single group code
    bin_counts = np.array([len(rule[k]) - 1 for k in keys], dtype=np.int64)
    strides = np.empty(n_keys, dtype=np.int64)
    total = 1
    for i in range(n_keys):
        strides[i] = total
        total *= bin_counts[i]

    arrays = np.empty((n_rows, n_cols, n_keys), dtype=np.float64)
    if order:
        arrays[:, :, 0] = fast_rank(values_stack[0], rule[keys[0]])
        for i in range(1, n_keys):
            codes = np.zeros((n_rows, n_cols), dtype=np.int64)
            valid = np.ones((n_rows, n_cols), dtype=np.bool_)
            for j in range(i):
                b = arrays[:, :, j]
                m = b >= 1.0          # legal bins >= 1; NaN / 0 (no rank) excluded
                valid &= m
                codes += (np.where(m, b, 1.0).astype(np.int64) - 1) * strides[j]
            arrays[:, :, i] = _fast_rank_grouped(values_stack[i], codes, valid, rule[keys[i]])
    else:
        for i in range(n_keys):
            arrays[:, :, i] = fast_rank(values_stack[i], rule[keys[i]])

    joined = _join_labels(arrays, bin_counts)
    has_nan = np.isnan(arrays).any(axis=2)
    out = pd.DataFrame(joined, index=x.index, columns=cols)
    return out.where(~has_nan, other=np.nan)


def weight(df: pd.DataFrame, w_df: Optional[pd.DataFrame] = None,
           fillna: bool = True, pct: bool = True) -> pd.DataFrame:
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
        v = np.where(df.notnull().values, w_df.values, np.nan)   # float64 mask (orig used pd.NA -> object)
        if pct:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = v / np.nansum(v, axis=1, keepdims=True)
        return df * v
    else:
        if pct:
            return df.div(df.notnull().sum(axis=1), axis=0)
        else:
            return df


def portfolio(df_obj: pd.DataFrame, returns: pd.DataFrame,
              weight: Optional[pd.DataFrame] = None,
              shift: int = 1, roll: int = 1, fillna: bool = False) -> pd.DataFrame:
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
    returns = returns.sort_index()
    if roll > 1:
        returns = returns.rolling(roll).mean().shift(-(roll - 1))        

    df_obj = (fillna_func(df_obj.sort_index(), returns.index) if fillna else df_obj).shift(shift)
    df_obj = df_obj.reindex(index=returns.index, columns=returns.columns)   # same alignment as pd.concat
    if weight is not None:
        weight = (fillna_func(weight, returns.index) if fillna else weight).reindex_like(returns)

    r = np.ascontiguousarray(returns.values, dtype=np.float64)   # (n_rows, n_cols)
    n_rows, n_cols = r.shape
    lab = df_obj.values.ravel()                                  # labels (str/float/...), NaN = missing

    codes, uniques = pd.factorize(lab)
    n_codes = len(uniques)
    if n_codes == 0:
        return pd.DataFrame(dtype='float64')

    # column order: sorted labels (matches groupby-sort + sort_index(axis=1))
    sorted_uniques = np.sort(uniques)
    pos = np.searchsorted(sorted_uniques, uniques)               # appearance-order -> sorted-order

    lab_ok = codes >= 0
    rf = r.ravel()
    valid = lab_ok & ~np.isnan(rf)
    if weight is not None:
        wf = np.ascontiguousarray(weight.values, dtype=np.float64).ravel()
        valid &= ~np.isnan(wf)
        rw = rf * wf

    row_ids = np.repeat(np.arange(n_rows), n_cols)
    key = row_ids[valid] * n_codes + pos[codes[valid]]
    if weight is not None:
        num = np.bincount(key, weights=rw[valid], minlength=n_rows * n_codes)
        den = np.bincount(key, weights=wf[valid], minlength=n_rows * n_codes)
    else:
        num = np.bincount(key, weights=rf[valid], minlength=n_rows * n_codes)
        den = np.bincount(key, minlength=n_rows * n_codes)

    with np.errstate(divide='ignore', invalid='ignore'):
        res = num / den
    res = res.reshape(n_rows, n_codes)

    # pandas>=2.1 stack() keeps label-only cells, so a date stays in the output
    # as long as it has >= 1 valid group label (its returns may be all NaN)
    keep = lab_ok.reshape(n_rows, n_cols).any(axis=1)
    res = res[keep]
    cols = pd.Index(sorted_uniques, name='portfolio')
    return pd.DataFrame(res, index=returns.index[keep], columns=cols)


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

def d_cut(
    df_obj: pd.DataFrame,
    count: Union[int, List[int]],
    max_count: Union[int, List[int]],
    delay: int
) -> pd.DataFrame:
    """
    ===========================================================================
    Dynamic top-N selection with a mandatory holding period and buffer zone
    to reduce turnover.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input factor DataFrame.
    count : Union[int, List[int]]
        The maximum number of picks, either constant or per-period.
    max_count : Union[int, List[int]]
        The buffer count allowed beyond count, either constant or per-period.
    delay : int
        The minimum holding periods for newly selected assets.

    Returns
    -------
    pd.DataFrame
        DataFrame with ranks for selected assets and 0 otherwise.
    ---------------------------------------------------------------------------
    带强制持有期和缓冲区的动态 Top-N 选择, 以降低换手率.

    参数
    ----
    df_obj : pd.DataFrame
        输入的因子 DataFrame.
    count : Union[int, List[int]]
        最大选择数量, 可为常量或按周期变化的列表.
    max_count : Union[int, List[int]]
        count 之上允许的缓冲区数量, 可为常量或按周期变化的列表.
    delay : int
        新入选资产的最小持有周期数.

    返回
    ----
    pd.DataFrame
        选中资产返回排名值, 其余为 0 的 DataFrame.
    ---------------------------------------------------------------------------
    """
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
    fillna : Union[int, float]
        The value used to fill NaN cells before weighting. Default is 0.

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
    fillna : Union[int, float]
        加权前用于填充 NaN 单元格的值. 默认为 0.

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
    
    
