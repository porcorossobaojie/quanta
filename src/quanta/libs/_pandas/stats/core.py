# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:29:21 2026

@author: Porco Rosso
"""
import os
os.environ.pop('NUMBA_DISABLE_JIT', None)
os.environ['NUMBA_NUM_THREADS'] = f'{os.cpu_count()}'
os.environ['NUMBA_THREADING_LAYER'] = 'tbb'
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from functools import partial
import scipy as sp
import statsmodels.api as sm
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

from quanta.libs.utils import dict_to_dataclass
import numba
from numba import njit, prange
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ['standard', 'OLS', 'const', 'neutral', 'expose']

def standard(
    df_obj: Union[pd.Series, pd.DataFrame],
    method: str = 'gauss',
    rank: Tuple[Optional[float], Optional[float]] = (-5, 5),
    axis: Optional[int] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    ===========================================================================
    Standardizes a Series or DataFrame using Gaussian or uniform ranking
    methods.

    Parameters
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        The input data to standardize.
    method : str
        Standardization method: 'gauss' (Gaussian/Z-score) or 'uniform'
        (ranking to a range). Default is 'gauss'.
    rank : Tuple[Optional[float], Optional[float]]
        The clipping range for 'gauss' or mapping range for 'uniform'.
        Default is (-5, 5).
    axis : Optional[int]
        Axis along which to standardize. Default is None (0 for Series,
        1 for DataFrame).

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        The standardized data.
    ---------------------------------------------------------------------------
    使用高斯或均匀排名方法对 Series 或 DataFrame 进行标准化.

    参数
    ----
    df_obj : Union[pd.Series, pd.DataFrame]
        要标准化的输入数据.
    method : str
        标准化方法: 'gauss' (高斯/Z-score) 或 'uniform' (排名映射到指定范围).
        默认为 'gauss'.
    rank : Tuple[Optional[float], Optional[float]]
        'gauss' 的裁剪范围或 'uniform' 的映射范围. 默认为 (-5, 5).
    axis : Optional[int]
        标准化的轴. 默认为 None (Series 为 0, DataFrame 为 1).

    返回
    ----
    Union[pd.Series, pd.DataFrame]
        标准化后的数据.
    ---------------------------------------------------------------------------
    """
    axis = 0 if axis is None else axis
    if method == 'gauss':
        y = df_obj.sub(df_obj.mean(axis=axis), axis=0 if axis or isinstance(df_obj, pd.Series) else 1).div(df_obj.std(axis=axis), axis=0 if axis or isinstance(df_obj, pd.Series) else 1)
        y = y.clip(*rank)
    elif method == 'uniform':
        y = df_obj.rank(pct=True, axis=axis)
        rank = (0 if rank[0] is None else rank[0], 1 if rank[1] is None else rank[1])
        y = y * (rank[1] - rank[0]) + rank[0]
    else:
        y = df_obj
    return y

def OLS(
    df_obj: pd.DataFrame,
    const: bool = True,
    roll: Optional[int] = None,
    min_periods: Optional[int] = None,
    dropna: bool = True,
    keys: Tuple[int, int] = (0, -1),
    returns: type = dict,
    weight: Optional[pd.DataFrame] = None
) -> Union[Dict[Any, sm.regression.linear_model.RegressionResultsWrapper], List[sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    ===========================================================================
    Performs Ordinary Least Squares (OLS) or Weighted Least Squares (WLS)
    regression, supporting rolling windows.

    Parameters
    ----------
    df_obj : pd.DataFrame
        Input data: first column is dependent, others are independent.
    const : bool
        If True, adds a constant term to the independent variables.
        Default is True.
    roll : Optional[int]
        Rolling window size for regression. Default is None (full data).
    min_periods : Optional[int]
        Minimum observations required in window. Default is None (0).
    dropna : bool
        If True, drops rows with NaNs before regression. Default is True.
    keys : Tuple[int, int]
        Indicator for result dictionary keys: (0 for index, 1 for columns),
        followed by position. Default is (0, -1).
    returns : type
        Return type: dict or list. Default is dict.
    weight : Optional[pd.DataFrame]
        Weights for Weighted Least Squares. Default is None.

    Returns
    -------
    Union[Dict, List]
        Regression results as a dictionary or list.
    ---------------------------------------------------------------------------
    执行普通最小二乘 (OLS) 或加权最小二乘 (WLS) 回归, 支持滚动窗口.

    参数
    ----
    df_obj : pd.DataFrame
        输入数据: 第一列为因变量, 其他为自变量.
    const : bool
        如果为 True, 则向自变量添加常数项. 默认为 True.
    roll : Optional[int]
        回归的滚动窗口大小. 默认为 None (全数据).
    min_periods : Optional[int]
        窗口中所需的最小观测数. 默认为 None (0).
    dropna : bool
        如果为 True, 则在回归前删除包含 NaN 的行. 默认为 True.
    keys : Tuple[int, int]
        结果字典键的指示器: (0 表示索引, 1 表示列), 后跟位置. 默认为 (0, -1).
    returns : type
        返回类型: dict 或 list. 默认为 dict.
    weight : Optional[pd.DataFrame]
        加权最小二乘的权重. 默认为 None.

    返回
    ----
    Union[Dict, List]
        字典或列表形式的回归结果.
    ---------------------------------------------------------------------------
    """
    df = df_obj.copy()
    roll = len(df) if roll is None or roll > len(df) else roll
    min_periods = 0 if min_periods is None else min_periods
    df.insert(1, 'const', 1) if const is True else None
    dic = {}

    for i in range(len(df) - roll + 1):
        y = df.iloc[i: i + roll]
        w = weight.iloc[i: i + roll] if weight is not None else 1.0
        key = y.index[keys[1]] if keys[0] == 0 else y.columns[keys[1]]
        if len(y.dropna()) >= min_periods:
            dic[key] = sm.WLS(y.iloc[:, 0].astype(float), y.iloc[:, 1:].astype(float), weights=w, missing='drop').fit()
        elif dropna is False:
            dic[key] = None
        if returns is dict:
            return dic
    if isinstance(returns, dict):
        return dic
    else:
        results = list(dic.values())
        if len(results) == 1:
            return results[0]
        return results


def const(
    df_obj: Union[pd.Series, pd.DataFrame],
    columns: Optional[List[Any]] = None,
    prefix: Optional[Union[str, List[str]]] = None,
    sep: str = ''
) -> pd.DataFrame:
    """
    ===========================================================================
    Creates dummy/indicator variables for categorical data.

    Parameters
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        Input data containing categorical values.
    columns : Optional[List[Any]]
        Specific columns to convert. Default is None (all columns).
    prefix : Optional[Union[str, List[str]]]
        Prefix to prepend to dummy column names. Default is None.
    sep : str
        Separator between prefix and dummy names. Default is ''.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the dummy variables.
    ---------------------------------------------------------------------------
    为分类数据创建虚拟/指示变量.

    参数
    ----
    df_obj : Union[pd.Series, pd.DataFrame]
        包含分类值的输入数据.
    columns : Optional[List[Any]]
        要转换的特定列. 默认为 None (所有列).
    prefix : Optional[Union[str, List[str]]]
        添加到虚拟列名的前缀. 默认为 None.
    sep : str
        前缀与虚拟名之间的分隔符. 默认为 ''.

    返回
    ----
    pd.DataFrame
        包含虚拟变量的 DataFrame.
    ---------------------------------------------------------------------------
    """
    return pd.get_dummies(df_obj, prefix=prefix, prefix_sep=sep, columns=columns)

@njit(parallel=True, cache=True, nopython=True)
def fast_wls_w3d(data_3d, weights, l2=0):
    n_cols = data_3d.shape[0]
    n_slices = data_3d.shape[1]
    k_samples = data_3d.shape[2]
    betas = np.full((n_slices, k_samples - 1), np.nan)
    multi_w1 = (weights.shape[1] > 1)
    multi_w2 = (weights.shape[2] > 1)
    l2_bool = (l2 == 0)
    for i in prange(n_slices):
        slice_data = data_3d[:, i, :]
        valid_mask = np.ones(n_cols, dtype=np.bool_)
        for c in range(n_cols):
            for k in range(k_samples):
                if np.isnan(slice_data[c, k]):
                    valid_mask[c] = False
                    break

        if np.sum(valid_mask) > (2 * (k_samples - 1)):
            if multi_w1:
                w_sub = weights[valid_mask, i]
            else:
                w_sub = weights[valid_mask, 0]
            sqrt_w = np.sqrt(w_sub)
            y = slice_data[valid_mask, 0] * sqrt_w[:, 0]
            if multi_w2:
                x = slice_data[valid_mask, 1:] * sqrt_w[:, 1:]
            else:
                x = slice_data[valid_mask, 1:]
                for col in range(x.shape[1]):
                    x[:, col] = x[:, col] * sqrt_w[:, 0]
            xTx = x.T @ x
            xTy = x.T @ y
            if l2_bool:
                betas[i] = np.linalg.solve(xTx, xTy)
            else:
                for j in range(xTx.shape[0]):
                    xTx[j,j] = l2 + xTx[j,j]
                betas[i] = np.linalg.solve(xTx, xTy)
    return betas

@njit(parallel=True, cache=True, nopython=True)
def fast_ols_3d(data_3d, weights, l2=0):
    n_cols = data_3d.shape[0]
    n_slices = data_3d.shape[1]
    k_samples = data_3d.shape[2]
    betas = np.full((n_slices, k_samples - 1), np.nan)
    l2_bool = (l2 == 0)
    for i in prange(n_slices):
        slice_data = data_3d[:, i, :]
        valid_mask = np.ones(n_cols, dtype=np.bool_)
        for c in range(n_cols):
            for k in range(k_samples):
                if np.isnan(slice_data[c, k]):
                    valid_mask[c] = False
                    break
        if np.sum(valid_mask) > (2 * (k_samples - 1)):
            y = slice_data[valid_mask, 0]
            x = slice_data[valid_mask, 1:]
            xTx = x.T @ x
            xTy = x.T @ y

            if l2_bool:
                betas[i] = np.linalg.solve(xTx, xTy)
            else:
                for j in range(xTx.shape[0]):
                    xTx[j,j] = l2 + xTx[j,j]
                betas[i] = np.linalg.solve(xTx, xTy)
    return betas

@njit(parallel=True, cache=True, nopython=True, fastmath=True)
def fast_std(arr):
    n_rows, n_cols, n_depth = arr.shape
    result = np.full((n_rows, n_cols), np.nan, dtype=arr.dtype)
    for i in prange(n_rows):
        for j in range(n_cols):
            sum_sq = 0.0
            count = 0
            # 单次遍历，只累加非 NaN 的平方
            for k in range(n_depth):
                val = arr[i, j, k]
                if not np.isnan(val):
                    sum_sq += val * val
                    count += 1
            if count > 1:
                result[i, j] = np.sqrt(sum_sq / count)
    return result

def fast_wls(data_3d, weights, l2):
    l2_bool = (l2 == 0)
    def core_func(data_2d, w, l2):
        not_nan = ~np.isnan(data_2d).any(axis=1)
        m = data_2d[not_nan, :]
        if m.shape[0] > m.shape[1] * 2:
            if w is not None:
                w = w[not_nan, :] if w.ndim == 2 else w[not_nan][:, np.newaxis]
                m = m * w
            if l2_bool:
                x = np.linalg.solve(m[:, 1:].T @ m[:, 1:], m[:, 1:].T @ m[:, 0])
            else:
                x = np.linalg.solve(m[:, 1:].T @ m[:, 1:] + l2 * np.eye(m.shape[1]-1), m[:, 1:].T @ m[:, 0])
        else:
            x = np.array([np.nan] * (m.shape[1] - 1))
        return x
    if weights is not None and data_3d.shape[:2] == weights.shape[:2]:
        partial_func = partial(core_func, l2=l2)
        x = list(map(core_func, data_3d, weights))
    else:
        partial_func = partial(core_func, w=weights, l2=l2)
        x = list(map(partial_func, data_3d))
    params = np.array(x)
    return params

def neutral(
    df_obj: pd.DataFrame,
    const: bool = True,
    neu_axis: int = 1,
    periods: Optional[int] = None,
    w: Optional[np.ndarray] = None,
    l2: float = 0,
    resid: bool = False,
    dtype = None,
    **key_factors: pd.DataFrame
) -> Any:
    # 数据对齐
    dtype = 'float32' if periods is not None else ('float64' if dtype is None else dtype)
    l2 = np.array(l2).astype(dtype)
    raw = (
        {'__y__': df_obj} |
        ({'const': pd.DataFrame().reindex_like(df_obj).fillna(1)} if const else {}) |
        {i:j.reindex_like(df_obj) for i,j in key_factors.items()}
    )
    # 权重调整成3d并对齐(jit传入的数组的维度必须一致,不能一个3d一个1d)
    if isinstance(w, pd.DataFrame):
        w = w.reindex_like(df_obj).fillna(0).values[np.newaxis, :, :].astype(dtype)
    else:
        if w is not None:
            if w.ndim == 1:
                w = np.array(w)[np.newaxis, :, np.newaxis] if neu_axis==0 else np.array(w)[np.newaxis, np.newaxis, :].astype(dtype)
            elif w.ndim == 2:
                w = np.array(w)[np.newaxis, :, :].astype(dtype)

    # 根据neu_axis及逆行转置
    trans_dic = {
        (0, 3, 0): [1, 2, 0],
        (1, 3, 0): [2, 1, 0],
        (0, 3, 1): [2, 1, 0],
        (1, 3, 1): [1, 2, 0],
    }
    vals = np.array([i.values for i in raw.values()], dtype=dtype)
    vals = vals.transpose(*trans_dic.get((neu_axis, vals.ndim, periods is None)))
    w = w.transpose(*trans_dic.get((neu_axis, w.ndim, periods is None))) if w is not None else w

    # 根据权重情况决定调用函数

    labels = {
        1: {'index': df_obj.columns, 'columns':df_obj.index},
        0: {'index': df_obj.index, 'columns':df_obj.columns},
    }

    if periods is not None:
        func_dic = {
            None: fast_ols_3d,
            3:fast_wls_w3d
        }
        func = func_dic.get(w if w is None else w.ndim, fast_ols_3d)
        params = np.full((vals.shape[0], vals.shape[1], vals.shape[2]-1), np.nan, dtype=dtype)
        for p in range(periods, vals.shape[0] + 1):
            data_3d = vals[p-periods:p, :, :]
            weights = w[p-periods:p, :, :] if (w is not None and w.shape[0] != periods) else w
            params[p-1, :, :] = func(data_3d, weights, l2)
        if resid:
            resids = (
                sliding_window_view(vals[:, :, 0], window_shape=periods, axis=0) -
                np.einsum(
                    "rckw, rck -> rcw",
                    sliding_window_view(vals[:, :, 1:], window_shape=periods, axis=0), params[periods-1:], optimize=True)
                )
            resids = resids.transpose(1, 0, 2) if neu_axis == 1 else resids
            resids = dict_to_dataclass({'index': df_obj.index[-resids.shape[0]:], 'columns': df_obj.columns[-resids.shape[1]:], 'values': resids, 'std': fast_std}, name='resid')
        params = {
            j: pd.DataFrame(
                params[:, :, i],
                index=labels[neu_axis]['index'],
                columns=labels[neu_axis]['columns']
            )
            for i,j in enumerate(list(raw.keys())[1:])
            }
        if neu_axis == 1:
            params = {i:j.T for i,j in params.items()}
        params = pd.concat(params, axis=1)
    else:
        params = fast_wls(vals, w, l2)
        if resid:
            resids = vals[:, :, 0] - np.einsum("rck, rk -> rc", vals[:, :, 1:], params)
            resids = resids if neu_axis == 1 else resids.T
            resids = pd.DataFrame(resids, index=df_obj.index, columns=df_obj.columns)
        params = pd.DataFrame(params, index=df_obj.axes[int(not bool(neu_axis))], columns=list(raw.keys())[1:])

    result = dict_to_dataclass({'params':params, 'resid': resids if resid else None}, name='Neutral')
    return result

def expose(df_obj, *xs, limit=0.05, max_iter=2):
    df_neu = df_obj.stats.neutral(**{i.__str__():j[0] for i,j in enumerate(xs)}).resid
    def expose_1dim(y, x, v):
        y = y.stats.neutral(fac=x).resid
        y_std = y.std(axis=1)
        beta = (y_std * v) / (x.std(axis=1) * (1 - v ** 2))
        hat = y + x.mul(beta, axis=0)
        return hat
    itered = 0
    _xs = xs
    while (itered <= max_iter) and len(_xs):
        for i in _xs:
            df_neu = expose_1dim(df_neu, i[0], i[1])
        check = {i:np.abs(df_neu.corrwith(j[0], axis=1).mean()-j[1]) for i,j in enumerate(_xs)}
        _xs = [_xs[i] for i,j in check.items() if j > limit]
        itered += 1
    for i,j in enumerate(xs):
        print(f"factor_{i+1}: expose -> <{round(df_neu.corrwith(j[0], axis=1).mean(), 4)}> with hope <{j[1]}>")
    return df_neu




'''
import quanta
from quanta import flow
df_obj = flow.astock('ret')
key_factors = {'amount':flow.astock('free_turnover')}
const: bool = True
neu_axis: int = 0
periods: Optional[int] = 252
#w: Optional[np.ndarray] = None
resid: bool = True
#w = quanta.faclib.barra.us4.size()
w = pd.tools.halflife(252, 63)
p = periods
dtype=None
g1 = df_obj.stats.neutral(amount=flow.astock('free_turnover'), w=w, neu_axis=0, periods=252)
'''
