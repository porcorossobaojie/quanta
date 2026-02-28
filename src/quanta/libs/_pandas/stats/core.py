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

from ...utils import flatten_list

__all__ = ['standard', 'OLS', 'const', 'neutral']

def standard(
    df_obj: Union[pd.Series, pd.DataFrame],
    method: str = 'gauss',
    rank: Tuple[Optional[float], Optional[float]] = (-5, 5),
    axis: Optional[int] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    ===========================================================================

    Standardizes a Series or DataFrame using specified method.

    This function applies Gaussian or uniform standardization to the input data.

    ---------------------------------------------------------------------------

    使用指定方法标准化 Series 或 DataFrame。

    此函数对输入数据应用高斯或均匀标准化。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        The input Series or DataFrame to standardize.
    method : str, optional
        Standardization method ('gauss' for Gaussian, 'uniform' for uniform),
        by default 'gauss'.
    rank : Tuple[Optional[float], Optional[float]], optional
        The rank range for uniform standardization, by default (-5, 5).
    axis : Optional[int], optional
        Axis along which to standardize, by default None (0 for Series, 1 for DataFrame).

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        要标准化的输入 Series 或 DataFrame。
    method : str, optional
        标准化方法（'gauss' 为高斯，'uniform' 为均匀），默认为 'gauss'。
    rank : Tuple[Optional[float], Optional[float]], optional
        均匀标准化的排名范围，默认为 (-5, 5)。
    axis : Optional[int], optional
        标准化轴，默认为 None（Series 为 0，DataFrame 为 1）。

    ---------------------------------------------------------------------------

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        The standardized Series or DataFrame.

    ---------------------------------------------------------------------------

    返回
    -------
    Union[pd.Series, pd.DataFrame]
        标准化后的 Series 或 DataFrame。

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

    Performs Ordinary Least Squares (OLS) regression.

    This function fits an OLS model, optionally with a constant, rolling window,
    and weights.

    ---------------------------------------------------------------------------

    执行普通最小二乘 (OLS) 回归。

    此函数拟合 OLS 模型，可选择包含常数项、滚动窗口和权重。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame, where the first column is the dependent variable
        and subsequent columns are independent variables.
    const : bool, optional
        If True, adds a constant to the independent variables, by default False.
    roll : Optional[int], optional
        The rolling window size for regression, by default None (full data).
    min_periods : Optional[int], optional
        Minimum number of observations in window required to have a value,
        by default None (0).
    dropna : bool, optional
        If True, drops rows with NaN values before regression, by default True.
    keys : Tuple[int, int], optional
        Tuple indicating how to get the key for the results dictionary.
        (0 for index, 1 for columns), (index/column position), by default (0, -1).
    returns : type, optional
        Type of return value (dict or list), by default dict.
    weight : Optional[pd.DataFrame], optional
        Weights for Weighted Least Squares (WLS), by default None.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        输入 DataFrame，其中第一列是因变量，后续列是自变量。
    const : bool, optional
        如果为 True，则向自变量添加一个常数项，默认为 False。
    roll : Optional[int], optional
        回归的滚动窗口大小，默认为 None（全数据）。
    min_periods : Optional[int], optional
        窗口中需要有值的最小观察数，默认为 None (0)。
    dropna : bool, optional
        如果为 True，则在回归前删除包含 NaN 值的行，默认为 True。
    keys : Tuple[int, int], optional
        元组，指示如何获取结果字典的键。
        （0 表示索引，1 表示列），（索引/列位置），默认为 (0, -1)。
    returns : type, optional
        返回值类型（dict 或 list），默认为 dict。
    weight : Optional[pd.DataFrame], optional
        加权最小二乘 (WLS) 的权重，默认为 None。

    ---------------------------------------------------------------------------

    Returns
    -------
    Union[Dict[Any, sm.regression.linear_model.RegressionResultsWrapper], List[sm.regression.linear_model.RegressionResultsWrapper]]
        A dictionary or list of OLS regression results.

    ---------------------------------------------------------------------------

    返回
    -------
    Union[Dict[Any, sm.regression.linear_model.RegressionResultsWrapper], List[sm.regression.linear_model.RegressionResultsWrapper]]
        OLS 回归结果的字典或列表。

    ---------------------------------------------------------------------------
    """
    df = df_obj.copy()
    roll = len(df) if roll is None or roll > len(df) else roll
    min_periods = 0 if min_periods is None else min_periods
    df.insert(1, 'const', 1) if const is True else None
    dic = {}

    for i in range(len(df) - roll + 1):
        y = df.iloc[i: i + roll]
        w = weight.iloc[i : i + roll] if weight is not None else 1.0
        key = y.index[keys[1]] if keys[0] == 0 else y.columns[keys[1]]
        if len(y.dropna()) >= min_periods:
            dic[key] = sm.WLS(y.iloc[:, 0].astype(float), y.iloc[:,1:].astype(float), weights=w, missing='drop').fit()
        elif dropna == False:
            dic[key] = None
        if returns is dict:
            return dic
    if isinstance(returns, dict):
        return dic
    else:
        dic = list(dic.values())
        if len(dic) == 1:
            dic = dic[0]
        return dic

def const(
    df_obj: Union[pd.Series, pd.DataFrame],
    columns: Optional[List[Any]] = None,
    prefix: Optional[Union[str, List[str]]] = None,
    sep: str = ''
) -> pd.DataFrame:
    """
    ===========================================================================

    Creates dummy variables for categorical data.

    This function converts categorical data into dummy/indicator variables.

    ---------------------------------------------------------------------------

    为分类数据创建虚拟变量。

    此函数将分类数据转换为虚拟/指示变量。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        The input Series or DataFrame containing categorical data.
    columns : Optional[List[Any]], optional
        Columns to convert into dummy variables, by default None (all columns).
    prefix : Optional[Union[str, List[str]]], optional
        String to prepend to dummy column names, by default None.
    sep : str, optional
        Separator between prefix and column name, by default ''.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : Union[pd.Series, pd.DataFrame]
        包含分类数据的输入 Series 或 DataFrame。
    columns : Optional[List[Any]], optional
        要转换为虚拟变量的列，默认为 None（所有列）。
    prefix : Optional[Union[str, List[str]]], optional
        要添加到虚拟列名前的字符串，默认为 None。
    sep : str, optional
        前缀和列名之间的分隔符，默认为 ''。

    ---------------------------------------------------------------------------

    Returns
    -------
    pd.DataFrame
        DataFrame with dummy variables.

    ---------------------------------------------------------------------------

    返回
    -------
    pd.DataFrame
        包含虚拟变量的 DataFrame。

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

    Converts DataFrames into a 3D NumPy array for regression analysis.

    This function prepares data for multi-variate regression by stacking
    dependent and independent variables into a 3D array.

    ---------------------------------------------------------------------------

    将 DataFrame 转换为用于回归分析的 3D NumPy 数组。

    此函数通过将因变量和自变量堆叠到 3D 数组中来准备多元回归数据。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    target_df : pd.DataFrame
        The dependent variable DataFrame.
    const : bool, optional
        If True, adds a constant array, by default True.
    **kwargs : pd.DataFrame
        Independent variable DataFrames.

    ---------------------------------------------------------------------------

    参数
    ----------
    target_df : pd.DataFrame
        因变量 DataFrame。
    const : bool, optional
        如果为 True，则添加一个常数数组，默认为 True。
    **kwargs : pd.DataFrame
        自变量 DataFrame。

    ---------------------------------------------------------------------------

    Returns
    -------
    Any
        A custom object containing the 3D array and metadata.

    ---------------------------------------------------------------------------

    返回
    -------
    Any
        包含 3D 数组和元数据的自定义对象。

    ---------------------------------------------------------------------------
    """
    target_df = target_df.sort_index(axis=1).sort_index()
    dic = (
        {'target':target_df.values}
        | ({'const': np.ones_like(target_df)} if const else {})
        | {i:j.reindex_like(target_df).values for i,j in kwargs.items()}
    )
    x = type('array_3D',
             (),
             {'index': target_df.index,
              'columns': target_df.columns,
              'labels': list(dic.keys()),
              'values': np.array(list(dic.values())).transpose(1,2,0)
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

    Creates a rolling window view of a 3D NumPy array.

    This function generates a view of the input 3D array, where each element
    is a sub-array representing a rolling window.

    ---------------------------------------------------------------------------

    创建 3D NumPy 数组的滚动窗口视图。

    此函数生成输入 3D 数组的视图，其中每个元素都是表示滚动窗口的子数组。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_3D : np.ndarray
        The input 3D NumPy array.
    periods : int
        The size of the rolling window.
    flatten : bool, optional
        If True, flattens the windowed array, by default False.

    ---------------------------------------------------------------------------

    参数
    ----------
    array_3D : np.ndarray
        输入的 3D NumPy 数组。
    periods : int
        滚动窗口的大小。
    flatten : bool, optional
        如果为 True，则展平窗口数组，默认为 False。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        The rolling window view of the array.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        数组的滚动窗口视图。

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
) -> np.ndarray:
    """
    ===========================================================================

    Performs least squares regression on a 2D array.

    This is an internal helper function for `_lstsq`.

    ---------------------------------------------------------------------------

    对 2D 数组执行最小二乘回归。

    这是 `_lstsq` 的内部辅助函数。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_2D : np.ndarray
        The input 2D array, with dependent variable in the first column.
    w : Optional[np.ndarray], optional
        Weights for weighted least squares, by default None.

    ---------------------------------------------------------------------------

    参数
    ----------
    array_2D : np.ndarray
        输入 2D 数组，因变量在第一列。
    w : Optional[np.ndarray], optional
        加权最小二乘的权重，默认为 None。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        Array of regression parameters.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        回归参数数组。

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
) -> np.ndarray:
    """
    ===========================================================================

    Applies least squares regression across a 3D array.

    This function iterates through slices of a 3D array and performs least
    squares regression on each slice.

    ---------------------------------------------------------------------------

    对 3D 数组应用最小二乘回归。

    此函数遍历 3D 数组的切片，并对每个切片执行最小二乘回归。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    array_3D : np.ndarray
        The input 3D array.
    neu_axis : int, optional
        Axis along which to perform regression (0 or 1), by default 1.
    w : Optional[np.ndarray], optional
        Weights for weighted least squares, by default None.

    ---------------------------------------------------------------------------

    参数
    ----------
    array_3D : np.ndarray
        输入的 3D 数组。
    neu_axis : int, optional
        执行回归的轴（0 或 1），默认为 1。
    w : Optional[np.ndarray], optional
        加权最小二乘的权重，默认为 None。

    ---------------------------------------------------------------------------

    Returns
    -------
    np.ndarray
        Array of regression parameters for each slice.

    ---------------------------------------------------------------------------

    返回
    -------
    np.ndarray
        每个切片的回归参数数组。

    ---------------------------------------------------------------------------
    """
    array_3D = array_3D.transpose(1, 0, 2) if neu_axis == 0 else array_3D

    if w is not None and array_3D.shape[:2] == w.shape:
        x = list(map(__lstsq, array_3D, w))
    else:
        partial_func = partial(__lstsq, w=w)
        x = list(map(partial_func, array_3D))
    params, t, p = zip(*x)
    params = np.array(params)
    t = np.array(t)
    p = np.array(p)
    return params, t, p

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

    Performs factor neutralization using linear regression.

    This function neutralizes a target DataFrame against specified factors,
    optionally over rolling periods.

    ---------------------------------------------------------------------------

    使用线性回归执行因子中性化。

    此函数根据指定因子对目标 DataFrame 进行中性化，可选择在滚动周期内进行。

    ---------------------------------------------------------------------------

    Parameters
    ----------
    df_obj : pd.DataFrame
        The target DataFrame to be neutralized.
    const : bool, optional
        If True, includes a constant term in the regression, by default True.
    neu_axis : int, optional
        Axis along which to perform neutralization (0 for rows, 1 for columns),
        by default 1.
    periods : Optional[int], optional
        Rolling window size for neutralization, by default None (full data).
    flatten : bool, optional
        If True, flattens the rolling window data, by default False.
    w : Optional[np.ndarray], optional
        Weights for weighted regression, by default None.
    resid : bool, optional
        If True, returns residuals; otherwise, returns parameters, by default True.
    **key_dfs : pd.DataFrame
        Factor DataFrames for neutralization.

    ---------------------------------------------------------------------------

    参数
    ----------
    df_obj : pd.DataFrame
        要中性化的目标 DataFrame。
    const : bool, optional
        如果为 True，则在回归中包含常数项，默认为 True。
    neu_axis : int, optional
        执行中性化的轴（0 为行，1 为列），默认为 1。
    periods : Optional[int], optional
        中性化的滚动窗口大小，默认为 None（全数据）。
    flatten : bool, optional
        如果为 True，则展平滚动窗口数据，默认为 False。
    w : Optional[np.ndarray], optional
        加权回归的权重，默认为 None。
    resid : bool, optional
        如果为 True，则返回残差；否则返回参数，默认为 True。
    **key_dfs : pd.DataFrame
        用于中性化的因子 DataFrame。

    ---------------------------------------------------------------------------

    Returns
    -------
    Any
        A custom object containing regression parameters and/or residuals.

    ---------------------------------------------------------------------------

    返回
    -------
    Any
        包含回归参数和/或残差的自定义对象。

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
            dic = {i: j.mul(self.params[i], axis=0) for i,j in kdfs.items()}
            dic = pd.concat(dic, axis=1)
            n_levels = list(range(1, dic.columns.nlevels))
            dic = dic.groupby(level=n_levels, axis=1).sum(min_count=1)
            if const:
                dic = dic.add(self.params['const'], axis=0)
            return dic

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
