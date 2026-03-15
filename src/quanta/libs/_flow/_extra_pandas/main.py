# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:50:12 2026

@author: Porco Rosso
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional, Union, List, Any
from quanta.config import settings
from quanta.libs.utils._decorator import doc_inherit
from .core import *
from .._cap.core import *

config = settings('flow')


class class_obj:
    """
    ===========================================================================
    Static utility class registered to Pandas for standalone flow extra
    operations.
    ---------------------------------------------------------------------------
    注册到 Pandas 的静态工具类, 用于独立的数据流额外操作.
    ---------------------------------------------------------------------------
    """
    merge = staticmethod(merge)
    Series = staticmethod(Series)
    DataFrame = staticmethod(DataFrame)
    unit = staticmethod(Unit)
    chain = staticmethod(Chain)

setattr(pd, config.extra_pandas_attrname, class_obj)


@pd.api.extensions.register_dataframe_accessor(config.extra_pandas_attrname)
class flow_extra():
    """
    ===========================================================================
    Pandas DataFrame accessor for extended data flow operations, including
    filtering, indexing, and backtesting.
    ---------------------------------------------------------------------------
    用于扩展数据流操作的 Pandas DataFrame 访问器, 包括过滤, 指数成份股和回测.
    ---------------------------------------------------------------------------
    """

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    @doc_inherit(listing)
    def listing(
        self,
        limit: int,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = listing(limit, portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]

    @doc_inherit(not_st)
    def not_st(
        self,
        value: int = 1,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = not_st(value, portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]

    @doc_inherit(statusable)
    def tradestatus(self, portfolio_type: Optional[str] = None) -> pd.DataFrame:
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = statusable(portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]

    @doc_inherit(filtered)
    def filtered(
        self,
        listing_limit: int = 126,
        drop_st: int = 1,
        tradestatus: bool = True,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(self._obj).fillna(False)
        return self._obj[x]

    @doc_inherit(index_members)
    def index_members(
        self,
        index_code: str,
        invert: bool = False
    ) -> pd.DataFrame:
        x = index_members(index_code).reindex_like(self._obj).fillna(False)
        x = ~x if invert else x
        return self._obj[x]

    @doc_inherit(label)
    def label(
        self,
        code: Optional[str] = None,
        label_df: Optional[pd.DataFrame] = None,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        portfolio_type = self._obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        x = label(code, label_df, portfolio_type)
        reindex_key = list(set(self._obj.columns.names) & set(x.columns.names))[0]
        df = self._obj.reindex(x.columns.get_level_values(reindex_key), axis=1)
        df.columns = x.columns
        x = x.reindex_like(df).fillna(False)
        return df[x]

    @doc_inherit(expand)
    def expand(self, portfolio_type: str = 'astock') -> pd.DataFrame:
        if portfolio_type == 'astock':
            target_df = label(self._obj.columns.name, portfolio_type=portfolio_type)
            x = expand(self._obj, target_df)
            return x
        else:
            raise ValueError('Warning: only support stock -- industry translate...')

    @doc_inherit(info)
    def info(
        self,
        column: str,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        return info(self._obj, column, portfolio_type)

    @lru_cache(maxsize=64)
    @doc_inherit(ic)
    def ic(
        self,
        listing_limit: int = 126,
        drop_st: int = 1,
        tradestatus: bool = True,
        portfolio_type: Optional[str] = None
    ) -> pd.Series:
        return ic(self._obj, listing_limit, drop_st, tradestatus, portfolio_type)

    @lru_cache(maxsize=64)
    @doc_inherit(ir)
    def ir(
        self,
        listing_limit: int = 126,
        drop_st: int = 1,
        tradestatus: bool = True,
        portfolio_type: Optional[str] = None
    ) -> pd.Series:
        x = self.ic(listing_limit, drop_st, tradestatus, portfolio_type)
        return ir(x)

    @lru_cache(maxsize=8)
    @doc_inherit(port)
    def port(
        self,
        listing_limit: int = 126,
        drop_st: int = 1,
        tradestatus: bool = True,
        portfolio_type: Optional[str] = None
    ) -> pd.DataFrame:
        x = port(self._obj, listing_limit, drop_st, tradestatus, portfolio_type)
        return x

    @lru_cache(maxsize=2)
    @doc_inherit(qtest)
    def test(
        self,
        shift: int = 0,
        high: Optional[str] = None,
        low: Optional[str] = None,
        avgprice: Optional[str] = None,
        trade_price: Optional[str] = None,
        settle_price: Optional[str] = None,
        limit: float = 0.01,
        trade_cost: float = 0.0015,
        portfolio_type: Optional[str] = None
    ) -> Any:
        df = self._obj if shift == 0 else self._obj.shift(shift).dropna(how='all')
        obj = qtest(df, high, low, avgprice, trade_price, settle_price, limit, trade_cost, portfolio_type)
        return obj

    @lru_cache(maxsize=2)
    @doc_inherit(qtest)
    def chain(
        self,
        cash=10000,
        trade_cost=True
    ):
        return Chain(self._obj, cash, trade_cost)

@pd.api.extensions.register_series_accessor(config.extra_pandas_attrname)
class flow_extra_series():
    """
    ===========================================================================
    Pandas Series accessor for extended data flow operations.
    ---------------------------------------------------------------------------
    用于扩展数据流操作的 Pandas Series 访问器.
    ---------------------------------------------------------------------------
    """

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    @doc_inherit(series_info)
    def info(
        self,
        column: str,
        portfolio_type: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        return series_info(self._obj, column, portfolio_type, **kwargs)

    @doc_inherit(day_shift)
    def day_shift(self, shift: int = 1, copy=True) -> pd.Series:
        return day_shift(self._obj, shift, copy)
