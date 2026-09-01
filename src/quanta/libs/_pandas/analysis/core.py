# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:22:04 2026

@author: Porco Rosso
"""
import numpy as np
import pandas as pd

__all__ = ['maxdown', 'sharpe', 'effective']

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(cache=True)
    def _maxdown_numba(vals, start_pos, end_pos, start_val, end_val, pct):
        n, k = vals.shape
        for j in range(k):
            peak = -np.inf
            best_dd, best_i, peak_at_best = np.inf, -1, np.nan
            for i in range(n):
                v = vals[i, j]
                if v != v:                       # NaN
                    continue
                if v > peak:
                    peak = v
                dd = v / peak - 1.0
                if dd < best_dd:
                    best_dd, best_i, peak_at_best = dd, i, peak
            if best_i == -1:                     # 全 NaN 列
                start_pos[j] = end_pos[j] = -1
                start_val[j] = end_val[j] = pct[j] = np.nan
                continue
            up_i = -1
            for i in range(best_i + 1):          # running max 首次达到谷底峰值的位置
                if vals[i, j] == peak_at_best:
                    up_i = i
                    break
            start_pos[j], end_pos[j] = up_i, best_i
            start_val[j], end_val[j] = peak_at_best, vals[best_i, j]
            pct[j] = best_dd

def _maxdown_numpy(vals):
    n, k = vals.shape
    nan = np.isnan(vals)
    fill = vals.copy(); fill[nan] = -np.inf
    np.maximum.accumulate(fill, axis=0, out=fill)          # fill = running peak
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = vals / fill - 1.0
    with np.errstate(invalid='ignore'):
        dd_min = np.nanmin(dd, axis=0)
    valid = ~np.isnan(dd_min); vi = np.nonzero(valid)[0]
    start_pos = np.full(k, -1, dtype=np.int64); end_pos = np.full(k, -1, dtype=np.int64)
    start_val = np.full(k, np.nan); end_val = np.full(k, np.nan)
    if vi.size:
        end_pos[vi] = np.argmax(dd[:, valid] == dd_min[valid][None, :], axis=0)
        peak_at_bottom = fill[end_pos[vi], vi]
        start_val[vi], end_val[vi] = peak_at_bottom, vals[end_pos[vi], vi]
        for j in vi:                                        # 峰值单调 -> 二分
            start_pos[j] = np.searchsorted(fill[:, j], peak_at_bottom[j], side='left')
    return start_pos, end_pos, start_val, end_val, dd_min

def maxdown(df_obj: pd.DataFrame, iscumprod: bool = True) -> pd.DataFrame:
    """
    ===========================================================================
    Calculates the maximum drawdown and related statistics for each column
    in a DataFrame.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input data, typically cumulative returns or periodic returns.
    iscumprod : bool
        If False, the input is treated as periodic returns and converted to
        cumulative returns. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing start/end dates, values, and percentage
        drawdown for each column.
    ---------------------------------------------------------------------------
    计算 DataFrame 中每一列的最大回撤及相关统计指标.

    参数
    ----
    df_obj : pd.DataFrame
        输入数据, 通常为累计收益率或周期性收益率.
    iscumprod : bool
        如果为 False, 则将输入视为周期性收益并转换为累计收益. 默认为 True.

    返回
    ----
    pd.DataFrame
        包含每一列的起始/结束日期, 数值以及最大回撤百分比的 DataFrame.
    ---------------------------------------------------------------------------
    """
    if not iscumprod:
        x = df_obj.add(1, fill_value=0).cumprod()
        x[df_obj.isnull()] = pd.NA
    else:
        x = df_obj

    idx, cols = x.index, x.columns
    vals = x.to_numpy(dtype='float64', na_value=np.nan)     # pd.NA -> NaN; float64 时零拷贝
    k = vals.shape[1]

    if HAS_NUMBA:
        start_pos = np.empty(k, dtype=np.int64); end_pos = np.empty(k, dtype=np.int64)
        start_val = np.empty(k); end_val = np.empty(k); pct = np.empty(k)
        _maxdown_numba(vals, start_pos, end_pos, start_val, end_val, pct)
    else:
        start_pos, end_pos, start_val, end_val, pct = _maxdown_numpy(vals)

    start_dates = np.full(k, np.nan, dtype=object)
    end_dates = np.full(k, np.nan, dtype=object)
    ok = start_pos >= 0
    start_dates[ok] = idx[start_pos[ok]]
    end_dates[ok] = idx[end_pos[ok]]

    return pd.DataFrame(
        [start_dates, start_val, end_dates, end_val, pct],
        columns=cols,
        index=['start', 'start_value', 'end', 'end_value', 'pct']
    )




def maxdown_old(
    df_obj: pd.DataFrame,
    iscumprod: bool = True
) -> pd.DataFrame:
    """
    ===========================================================================
    Calculates the maximum drawdown and related statistics for each column
    in a DataFrame.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input data, typically cumulative returns or periodic returns.
    iscumprod : bool
        If False, the input is treated as periodic returns and converted to
        cumulative returns. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing start/end dates, values, and percentage
        drawdown for each column.
    ---------------------------------------------------------------------------
    计算 DataFrame 中每一列的最大回撤及相关统计指标.

    参数
    ----
    df_obj : pd.DataFrame
        输入数据, 通常为累计收益率或周期性收益率.
    iscumprod : bool
        如果为 False, 则将输入视为周期性收益并转换为累计收益. 默认为 True.

    返回
    ----
    pd.DataFrame
        包含每一列的起始/结束日期, 数值以及最大回撤百分比的 DataFrame.
    ---------------------------------------------------------------------------
    """
    if not iscumprod:
        x = df_obj.add(1, fill_value=0).cumprod()
        x[df_obj.isnull()] = pd.NA
    else:
        x = df_obj

    max_flow = x.expanding(min_periods=1).max()
    down_date = (x / max_flow).idxmin()
    max_down_series = x / max_flow - 1

    down_info = [(x.loc[j, i] if pd.notnull(j) else np.nan, max_down_series.loc[j, i] if pd.notnull(j) else np.nan) for i, j in down_date.items()]
    down_value, max_down_value = list(zip(*down_info))

    up_value = pd.Series([max_flow.loc[j, i] if pd.notnull(j) else np.nan for i, j in down_date.items()], index=down_date.index)
    up_date = max_flow[max_flow == up_value].idxmin()

    df = pd.DataFrame(
        [up_date.values, up_value.values, down_date.values, down_value, max_down_value],
        columns=df_obj.columns,
        index=['Maxdown_Start_Date', 'Maxdown_Start_Value', 'Maxdown_End_Date', 'Maxdown_End_Value', 'Maxdwon_Percent']
    )
    return df


def sharpe(
    df_obj: pd.DataFrame,
    iscumprod: bool = True,
    periods: int = 252
) -> pd.Series:
    """
    ===========================================================================
    Calculates the Sharpe ratio for each column in a DataFrame.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input data, typically periodic returns.
    iscumprod : bool
        If True, the input is treated as cumulative returns and converted
        to periodic returns. Default is True.
    periods : int
        The number of periods to annualize the Sharpe ratio. Default is 252.

    Returns
    -------
    pd.Series
        The annualized Sharpe ratio for each column.
    ---------------------------------------------------------------------------
    计算 DataFrame 中每一列的夏普比率.

    参数
    ----
    df_obj : pd.DataFrame
        输入数据, 通常为周期性收益率.
    iscumprod : bool
        如果为 True, 则将输入视为累计收益并转换为周期性收益. 默认为 True.
    periods : int
        用于年度化夏普比率的周期数. 默认为 252.

    返回
    ----
    pd.Series
        每一列的年度化夏普比率.
    ---------------------------------------------------------------------------
    """
    x = df_obj if not iscumprod else df_obj.pct_change(fill_method=None)
    y = x.mean() / x.std()

    if periods is not None:
        periods = min(len(x), periods)
        y = y * (periods ** 0.5)

    y.name = f'sharpe_ratio(periods={periods})'
    return y


def effective(df_obj: pd.DataFrame) -> pd.Series:
    """
    ===========================================================================
    Calculates the 'effective' indicator, typically used for group returns
    to evaluate factor linearity.

    This indicator quantifies factor quality by summing the sign-preserved
    squared differences of returns across ranked groups. Higher values
    indicate stronger monotonic relationships across factor bins.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame containing group returns (columns are groups).

    Returns
    -------
    pd.Series
        The calculated effective scores for each period.
    ---------------------------------------------------------------------------
    计算 'effective' 指标, 通常用于评估分组收益率的因子线性性.

    该指标通过对排名分组之间的收益率差值进行符号保留的平方加权求和来量化因子质
    量. 较高的数值表示因子分箱之间具有更强的单调关系.

    参数
    ----
    df_obj : pd.DataFrame
        包含分组收益率的输入 DataFrame (列为各组别).

    返回
    ----
    pd.Series
        每个周期计算得到的 effective 分数.
    ---------------------------------------------------------------------------
    """
    x = df_obj.diff(axis=1)
    x = np.sign(x) * x ** 2
    x = x.sum(axis=1)
    return x
