# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:29:28 2026

@author: Porco Rosso
"""

import statsmodels as sm
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from .core import *
from quanta.libs.utils._decorator import doc_inherit

MODULE_DIR = __name__.split('.')[-2]


@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas DataFrame accessor for statistical operations, including
    standardization, dummy variable creation, OLS regression, and
    factor neutralization.
    ---------------------------------------------------------------------------
    用于统计操作的 Pandas DataFrame 访问器, 包括标准化, 虚拟变量创建, OLS 回归和
    因子中性化.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @doc_inherit(standard)
    def standard(
        self,
        method: str = 'gauss',
        rank: Tuple[Optional[float], Optional[float]] = (-5, 5),
        axis: int = 1
    ) -> pd.DataFrame:
        x = standard(self._obj, method=method, rank=rank, axis=axis)
        return x

    @doc_inherit(const)
    def const(
        self,
        columns: Optional[List[Any]] = None,
        prefix: Optional[Union[str, List[str]]] = None,
        sep: str = ''
    ) -> pd.DataFrame:
        return const(self._obj, columns=columns, prefix=prefix, sep=sep)

    @doc_inherit(OLS)
    def OLS(
        self,
        const: bool = True,
        roll: Optional[int] = None,
        min_periods: Optional[int] = None,
        dropna: bool = True,
        keys: Tuple[int, int] = (0, -1),
        returns: type = list,
        weight: Optional[pd.DataFrame] = None
    ) -> Union[Dict[Any, sm.regression.linear_model.RegressionResultsWrapper], List[sm.regression.linear_model.RegressionResultsWrapper]]:
        return OLS(self._obj, const=const, roll=roll, min_periods=min_periods, dropna=dropna, keys=keys, returns=returns, weight=weight)

    @doc_inherit(neutral)
    def neutral(
        self,
        const: bool = True,
        neu_axis: int = 1,
        periods: Optional[int] = None,
        flatten: bool = False,
        weight: Optional[np.ndarray] = None,
        resid: bool = True,
        **key_dfs: pd.DataFrame
    ) -> Any:
        return neutral(self._obj, const=const, neu_axis=neu_axis, periods=periods, flatten=flatten, w=weight, resid=resid, **key_dfs)


@pd.api.extensions.register_series_accessor(MODULE_DIR)
class main():
    """
    ===========================================================================
    Pandas Series accessor for statistical operations, including
    standardization and dummy variable creation.
    ---------------------------------------------------------------------------
    用于统计操作的 Pandas Series 访问器, 包括标准化和虚拟变量创建.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    @doc_inherit(standard)
    def standard(
        self,
        method: str = 'gauss',
        rank: Tuple[Optional[float], Optional[float]] = (-5, 5)
    ) -> pd.Series:
        return standard(self._obj, method=method, rank=rank, axis=None)

    @doc_inherit(const)
    def const(
        self,
        prefix: Optional[str] = None,
        sep: str = ''
    ) -> pd.DataFrame:
        return const(self._obj, prefix=prefix, sep=sep)
