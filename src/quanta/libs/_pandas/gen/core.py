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
    """table[c, j] = bin label of the (j+1)-th smallest value in a group of size c."""
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
    Rank each row into percentile bins defined by `rule`; NaN preserved.

    Same semantics as the original fast_rank, but uses a single argsort per row
    (rank of a value only depends on its sorted position) plus a lookup table
    instead of a second argsort and per-element searchsorted.
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
    For each row, rank `values` across the column axis within groups defined by
    `codes` (only where `valid` and value is not NaN), then map ranks to bins.

    Excluded cells are set to 0.0 -- replicating the original np.nansum(...,0)
    behaviour where an all-NaN slice sums to 0 instead of NaN. 0 is never a
    legal bin; downstream logic treats it as "no rank".
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
    'b0_b1_...' strings per cell; cells with any NaN key become '-1_...'
    (dropped later). Labels are packed into ints, deduplicated, formatted only
    for unique combos, then gathered back -- no per-element string work on the
    full matrix.
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
    Group / rank a DataFrame into percentile bins; see module docstring for
    details on the exact behavioural compatibility.
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
    
    
