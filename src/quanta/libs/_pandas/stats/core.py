# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:29:36 2026

@author: Porco Rosso
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from functools import partial
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

from quanta.libs.utils import flatten_list

__all__ = ['standard', 'OLS', 'const', 'neutral']


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


def _array_3D(
    target_df: pd.DataFrame,
    const: bool = True,
    **kwargs: pd.DataFrame
) -> Any:
    """
    ===========================================================================
    Converts DataFrames into a structured 3D NumPy array for efficient
    regression analysis.

    Parameters
    ----------
    target_df : pd.DataFrame
        The dependent variable DataFrame.
    const : bool
        If True, adds a constant term array. Default is True.
    **kwargs : pd.DataFrame
        Additional independent variable DataFrames.

    Returns
    -------
    Any
        A custom object containing the 3D values and metadata.
    ---------------------------------------------------------------------------
    将 DataFrame 转换为结构化 3D NumPy 数组, 用于高效的回归分析.

    参数
    ----
    target_df : pd.DataFrame
        因变量 DataFrame.
    const : bool
        如果为 True, 则添加常数项数组. 默认为 True.
    **kwargs : pd.DataFrame
        附加自变量 DataFrame.

    返回
    ----
    Any
        包含 3D 数值和元数据的自定义对象.
    ---------------------------------------------------------------------------
    """
    target_df = target_df.sort_index(axis=1).sort_index()
    dic = (
        {'target': target_df.values}
        | ({'const': np.ones_like(target_df)} if const else {})
        | {i: j.reindex_like(target_df).values for i, j in kwargs.items()}
    )
    x = type('array_3D',
             (),
             {'index': target_df.index,
              'columns': target_df.columns,
              'labels': list(dic.keys()),
              'values': np.array(list(dic.values())).transpose(1, 2, 0)
              }
    )
    return x


def _array_roll(
    array_3D: np.ndarray,
    periods: int,
    flatten: bool = False
) -> np.ndarray:
    """
    ===========================================================================
    Creates a rolling window view of a 3D NumPy array using stride tricks.

    Parameters
    ----------
    array_3D : np.ndarray
        The input 3D NumPy array.
    periods : int
        The rolling window size.
    flatten : bool
        If True, flattens the rolling windows into a 2D-like view.
        Default is False.

    Returns
    -------
    np.ndarray
        The windowed array view.
    ---------------------------------------------------------------------------
    使用步长技巧创建 3D NumPy 数组的滚动窗口视图.

    参数
    ----
    array_3D : np.ndarray
        输入的 3D NumPy 数组.
    periods : int
        滚动窗口大小.
    flatten : bool
        如果为 True, 则将滚动窗口展平为类 2D 视图. 默认为 False.

    返回
    ----
    np.ndarray
        分窗后的数组视图.
    ---------------------------------------------------------------------------
    """
    axis = 0
    new_shape = list(array_3D.shape)
    new_shape[axis] = [periods, new_shape[axis] - periods + 1, ]
    new_shape = tuple(flatten_list(new_shape))

    new_strides = list(array_3D.strides)
    new_strides[axis] = [array_3D.strides[axis], array_3D.strides[axis]]
    new_strides = tuple(flatten_list(new_strides))

    window = as_strided(array_3D, shape=new_shape, strides=new_strides)
    window = window.transpose(1, 0, 2, 3)
    if flatten:
        window = window.reshape(window.shape[0], -1, window.shape[-1])
    return window


def __lstsq(
    array_2D: np.ndarray,
    w: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    ===========================================================================
    Internal helper performing least squares regression on a 2D data slice.

    Parameters
    ----------
    array_2D : np.ndarray
        2D array slice (rows, features).
    w : Optional[np.ndarray]
        Weights for weighted least squares. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Tuple of (parameters, t-values, valid observation count).
    ---------------------------------------------------------------------------
    内部辅助函数, 对 2D 数据切片执行最小二乘回归.

    参数
    ----
    array_2D : np.ndarray
        2D 数据切片 (行, 特征).
    w : Optional[np.ndarray]
        加权最小二乘的权重. 默认为 None.

    返回
    ----
    Tuple[np.ndarray, np.ndarray, float]
        (参数, t值, 有效观测数) 元组.
    ---------------------------------------------------------------------------
    """
    not_nan = ~np.isnan(array_2D).any(axis=1)
    matrix = array_2D[not_nan, :]

    if w is not None:
        w = w[not_nan]

    y = matrix[:, 0]

    # Check for sufficient data points for regression
    if (matrix.shape[0] > matrix.shape[1] * 2) and matrix.shape[0] > 2:
        x = matrix[:, 1:]
        xT = x.T
        p = x.shape[0]
        if w is None:
            se = sp.linalg.pinv(xT.dot(x))
            t = np.diag(se) ** 0.5
            params = se.dot(xT).dot(y)
        else:
            se = sp.linalg.pinv((xT * w).dot(x))
            t = np.diag(se) ** 0.5
            params = se.dot(xT * w).dot(y)
    else:
        params = np.array([np.nan] * (matrix.shape[1] - 1))
        t = np.array([np.nan] * (matrix.shape[1] - 1))
        p = np.nan
    return params, t, p


def _lstsq(
    array_3D: np.ndarray,
    neu_axis: int = 1,
    w: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ===========================================================================
    Vectorized application of least squares regression across a 3D array.

    Parameters
    ----------
    array_3D : np.ndarray
        The input 3D data array.
    neu_axis : int
        The axis along which to perform regression. Default is 1.
    w : Optional[np.ndarray]
        Weights for weighted least squares. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Aggregated parameters, t-values, and counts.
    ---------------------------------------------------------------------------
    在 3D 数组上向量化应用最小二乘回归.

    参数
    ----
    array_3D : np.ndarray
        输入 3D 数据数组.
    neu_axis : int
        执行回归的轴. 默认为 1.
    w : Optional[np.ndarray]
        加权最小二乘的权重. 默认为 None.

    返回
    ----
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        汇总的参数, t值和计数.
    ---------------------------------------------------------------------------
    """
    array_3D = array_3D.transpose(1, 0, 2) if neu_axis == 0 else array_3D

    if w is not None and array_3D.shape[:2] == w.shape:
        x = list(map(__lstsq, array_3D, w))
    else:
        partial_func = partial(__lstsq, w=w)
        x = list(map(partial_func, array_3D))
    params, t, p = zip(*x)
    return np.array(params), np.array(t), np.array(p)


def neutral(
    df_obj: pd.DataFrame,
    const: bool = True,
    neu_axis: int = 1,
    periods: Optional[int] = None,
    flatten: bool = False,
    w: Optional[np.ndarray] = None,
    resid: bool = True,
    t: bool = True,
    **key_dfs: pd.DataFrame
) -> Any:
    """
    ===========================================================================
    Performs factor neutralization using multi-variate linear regression,
    optionally in rolling windows.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The target factor DataFrame to be neutralized.
    const : bool
        If True, includes a constant term. Default is True.
    neu_axis : int
        Axis for neutralization: 0 (cross-sectional per column), 1 (per row).
        Default is 1.
    periods : Optional[int]
        Rolling window size. Default is None (full data).
    flatten : bool
        If True, flattens rolling window data for regression.
        Default is False.
    w : Optional[np.ndarray]
        Weights for weighted regression. Default is None.
    resid : bool
        If True, returns residuals in the result object. Default is True.
    t : bool
        If True, returns t-values in the result object. Default is True.
    **key_dfs : pd.DataFrame
        Factors to neutralize against.

    Returns
    -------
    NeutralObj
        Custom result object with parameters, residuals, and statistics.
    ---------------------------------------------------------------------------
    使用多元线性回归执行因子中性化, 可选择在滚动窗口内进行.

    参数
    ----
    df_obj : pd.DataFrame
        要中性化的目标因子 DataFrame.
    const : bool
        如果为 True, 则包含常数项. 默认为 True.
    neu_axis : int
        中性化轴: 0 (每列横截面), 1 (每行). 默认为 1.
    periods : Optional[int]
        滚动窗口大小. 默认为 None (全数据).
    flatten : bool
        如果为 True, 则展平滚动窗口数据以进行回归. 默认为 False.
    w : Optional[np.ndarray]
        加权回归的权重. 默认为 None.
    resid : bool
        如果为 True, 则在结果对象中返回残差. 默认为 True.
    t : bool
        如果为 True, 则在结果对象中返回 t值. 默认为 True.
    **key_dfs : pd.DataFrame
        用于中性化的因子.

    返回
    ----
    NeutralObj
        包含参数, 残差和统计信息的自定义结果对象.
    ---------------------------------------------------------------------------
    """
    data_obj = _array_3D(df_obj, const, **key_dfs)
    values = data_obj.values

    if periods is not None:
        values = _array_roll(values, periods, flatten)
        if len(values.shape) == 4:
            if neu_axis == 0:
                values = values.transpose(0, 2, 1, 3)
                index = pd.MultiIndex.from_product([data_obj.index[periods - 1:], data_obj.columns], names=[df_obj.index.name, df_obj.columns.name])
            else:
                index = pd.MultiIndex.from_product([data_obj.index[periods - 1:], range(periods)], names=[df_obj.index.name, 'PERIOD'])
            values = values.reshape(values.shape[0] * values.shape[1], values.shape[2], values.shape[3])
        else:
            index = data_obj.index[periods - 1:]
    else:
        if neu_axis == 0:
            index = data_obj.columns
            columns = data_obj.index
            values = values.transpose(1, 0, 2)
        else:
            index = data_obj.index
            columns = data_obj.columns

    parameters, t_values, p = _lstsq(values, w=w)
    parameters = pd.DataFrame(parameters, index=index, columns=data_obj.labels[1:])
    t_df = pd.DataFrame(t_values, index=index, columns=data_obj.labels[1:])
    p_df = pd.Series(p, index=index, name='VAR_COUNT')

    class NeutralObj:
        def __init__(self, params, resid, t, p, r, adj_r):
            self.params = params
            self.resid = resid
            self.t = t
            self.var_count = p
            self.rsquared = r
            self.rsquared_adj = adj_r

        @property
        def p(self):
            df = pd.DataFrame(sp.stats.t.sf(self.t.abs(), self.var_count) * 2, index=self.t.index, columns=self.t.columns)
            return df

        def predict(self, const=True, **kdfs):
            dic = {i: j.mul(self.params[i], axis=0) for i, j in kdfs.items()}
            dic = pd.concat(dic, axis=1)
            n_levels = list(range(1, dic.columns.nlevels))
            dic = dic.groupby(level=n_levels, axis=1).sum(min_count=1)
            if const:
                dic = dic.add(self.params['const'], axis=0)
            return dic

    _resid = False
    if t:
        _resid = True

    if resid or _resid:
        if periods is None:
            resid_values = values[:, :, 0] - np.sum((values[:, :, 1:] * parameters.values[:, np.newaxis, :]), axis=-1)
            resid_df = pd.DataFrame(resid_values, index=index, columns=columns)
        else:
            resid_values = values[:, :, 0].astype(np.float16) - np.sum((values[:, :, 1:].astype(np.float16) * parameters.values[:, np.newaxis, :].astype(np.float16)), axis=-1)
            resid_df = pd.DataFrame(resid_values, index=index)
        r = (1 - resid_df.var(axis=1) / np.nanvar(values[:, :, 0], axis=1)).rename('r')
        adj_r = 1 - (1 - r) * (p_df - 1).values / (p_df - len(key_dfs)).rename('adj_r')
    else:
        resid_df = np.nan
        r = np.nan
        adj_r = np.nan

    if t:
        s = (
            (resid_df**2).sum(axis=1, min_count=1)
            /
            (resid_df.notnull().sum(axis=1) - len(key_dfs) - const)
        ) ** 0.5
        s = t_df.mul(s, axis=0)
        t_df = parameters / s
    else:
        t_df = np.nan

    if not resid:
        resid_df = np.nan
        r = np.nan
        adj_r = np.nan
    return NeutralObj(params=parameters, resid=resid_df, t=t_df, p=p_df, r=r, adj_r=adj_r)
