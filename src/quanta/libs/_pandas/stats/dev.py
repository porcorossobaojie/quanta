# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:29:21 2026

@author: Porco Rosso
"""


import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from functools import partial
import scipy as sp
import statsmodels.api as sm
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

from quanta.libs.utils import flatten_list
from numba import njit, prange

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

@njit(parallel=True)
def fast_wls_3d(data_3d, weights):
    n_slices = data_3d.shape[0]
    betas = np.full((n_slices, data_3d.shape[-1]), np.nan)
    w_is_3d = (weights.ndim == 3)

    for i in prange(n_slices):
        slice_data = data_3d[i]
        valid_mask = np.ones(k_samples, dtype=np.bool_)
        for r in range(k_samples):
            for c in range(n_cols):
                if np.isnan(slice_data[r, c]):
                    valid_mask[r] = False
                    break
        if np.sum(mask) > 2 * (data_3d.shape[-1] - 1): # 确保样本量足够
            y = slice_data[valid_mask, 0]
            X = slice_data[valid_mask, 1:] # 包含常数项列和x列
            if w_is_3d:
                w_sub = weights[i][valid_mask]
            else:
                w_sub = weights[valid_mask]
            y_sub = y[valid_mask]
            X_sub = X[valid_mask]
            sqrt_w = np.sqrt(w_sub)
            if w_is_3d:
                y_star = y_sub * sqrt_w[:, 0]
                X_star = X_sub * sqrt_w[:, 1:]
            else:
                y_star = y_sub * sqrt_w
                X_star = np.empty_like(X_sub)
                for col in range(X_sub.shape[1]):
                    X_star[:, col] = X_sub[:, col] * sqrt_w

            betas[i] = np.linalg.lstsq(X_star, y_star)[0]
    return betas

def neutral(
    df_obj: pd.DataFrame,
    *factors: pd.DataFrame,
    const: bool = True,
    neu_axis: int = 1,
    periods: Optional[int] = None,
    flatten: bool = False,
    w: Optional[np.ndarray] = None,
    resid: bool = True,
    **key_factors: pd.DataFrame
) -> Any:
    pass
    array_3d = {'__y__': df_obj} |({} in not const else {'const': pd.DataFrame().reindex_like())}|{str(i):j.reindex_like(df_obj) for i,j in enumerate(dfs)} |








