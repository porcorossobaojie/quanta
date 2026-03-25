# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:31:33 2026

@author: Porco Rosso
"""
from functools import lru_cache
import numexpr as ne
import numpy as np
import pandas as pd

from quanta import flow
from quanta.faclib._base.main import main as meta
#from .._base.main import main as meta
from quanta.config import settings
config = settings('factors')

class main(meta):
    """
    Base class for Barra factor calculations. | Barra因子计算基类.
    """
    finance = config.finance_keys

    @classmethod
    @lru_cache(maxsize=1)
    def size(cls) -> pd.DataFrame:
        """Calculate the size factor | 计算市值因子"""
        x = (flow.astock(cls.finance.val_mv) * 1e8).tools.log()
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def bm(cls) -> pd.DataFrame:
        """Calculate the book-to-market factor | 计算账面市值比因子"""
        #x = flow.astock(cls.finance.pb) ** -1
        total_assets = flow.astock.finance(cls.finance.total_assets, shift=2)
        mv = flow.astock(cls.finance.val_mv)
        x = (total_assets / mv / 1e8).stats.neutral(fac=cls.size()).resid
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def non_size(cls) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the non-linear size factor.

        Returns
        -------
        pd.DataFrame
            The calculated non-linear size factor.
        -----------------------------------------------------------------------
        计算非线性市值因子.

        返回
        ----
        pd.DataFrame
            计算得到的非线性市值因子.
        -----------------------------------------------------------------------
        """
        df = cls.size()
        df = (df ** 3).stats.neutral(me=df, weight= (df ** 0.5).values).resid.tools.log().stats.standard()
        return df

    @classmethod
    @lru_cache(maxsize=8)
    def _beta(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the beta-related metrics using weighted OLS.

        Parameters
        ----------
        periods : int
            The lookback period for calculation.
        halflife : int, optional
            The halflife for exponential weighting.
        bench : str
            The benchmark name to use.
        portfolio_type : str
            The type of portfolio (e.g., 'astock').

        Returns
        -------
        pd.DataFrame
            A MultiIndex DataFrame containing alpha, beta, and resid.
        -----------------------------------------------------------------------
        使用加权最小二乘法计算贝塔相关指标.

        参数
        ----
        periods : int
            计算使用的回看窗口.
        halflife : int, 可选
            指数加权的半衰期.
        bench : str
            使用的基准名称.
        portfolio_type : str
            组合类型 (例如 'astock').

        返回
        ----
        pd.DataFrame
            包含 alpha, beta 和 resid 的多重索引数据框.
        -----------------------------------------------------------------------
        """
        halflife = periods//4 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.trade.returns).fillna(0).astype('float32')
        entrade = ret.f.tradestatus().notnull()
        bench = cls.bench(bench)
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)[entrade].fillna(0).astype('float32')
        w = pd.tools.halflife(periods, halflife)[np.newaxis, :].astype('float32')

        roll = pd.concat({'ret':ret, 'w':entrade, 'bench':bench}, axis=1).rolling(periods).apply(lambda x: w @ x, raw=True)
        y = (ret - roll.ret /roll.w)
        x = (bench - roll.bench / roll.w)
        beta = (x * y).rolling(periods).apply(lambda x: w @ x, raw=True) / (x * x).rolling(periods).apply(lambda x: w @ x, raw=True)
        alpha = roll.ret / roll.w - beta * roll.bench / roll.w
        resid = (ret - alpha.bfill() - beta.bfill() * bench) ** 2
        resid = resid.rolling(periods).apply(lambda x: w @ x, raw=True)
        resid = resid / roll.w
        df = pd.concat({i:j.f.tradestatus(periods=periods, min_periods=halflife) for i,j in {'alpha':alpha, 'beta':beta, 'resid':resid}.items()}, axis=1)
        return df
    
    @classmethod
    @lru_cache(maxsize=8)
    def _beta2(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:

        halflife = periods//4 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.trade.returns).fillna(0).astype('float32')
        entrade = ret.f.tradestatus().notnull()
        bench = cls.bench(bench)
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)[entrade].fillna(0).astype('float32')
        w = pd.tools.halflife(periods, halflife).astype('float32')

        y_vals = pd.tools.array_roll(ret.values, periods)
        y_mean = np.einsum('twk, w -> tk', y_vals, w)
        x_vals = pd.tools.array_roll(bench.iloc[:, [0]].values, periods)[:, :, 0]
        x_mean = np.einsum('tw, w -> t', x_vals, w)[:, np.newaxis]
        x_vals = x_vals - x_mean
        y_vals = y_vals - y_mean[:, np.newaxis, :]
        txt = np.einsum('tw, w -> t', x_vals**2, w)[:, np.newaxis]
        beta = np.einsum('tw,w,twk->tk', x_vals, w, y_vals) / txt
        beta = pd.DataFrame(beta, columns=ret.columns, index=ret.index[periods-1:])
        alpha = y_mean - beta * x_mean
        b = beta.values[:, np.newaxis, :]
        a = alpha.values[:, np.newaxis, :]
        x = pd.tools.array_roll(bench.iloc[:, [0]].values, periods)[:, :, 0][:, :, np.newaxis]
        y_vals = ne.evaluate('(a + b * x - y_vals) ** 2', out=y_vals)
        y_vals = np.mean(y_vals, axis=1) ** 0.5
        resid = pd.DataFrame(y_vals, columns=ret.columns, index=ret.index[periods-1:])
        df = pd.concat({i:j.f.tradestatus(periods=periods, min_periods=halflife) for i,j in {'alpha':alpha, 'beta':beta, 'resid':resid}.items()}, axis=1)
        return df
        
        
    @classmethod
    def beta(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        """
        =======================================================================
        Extract the beta factor from beta-related metrics.

        Parameters
        ----------
        periods : int
            The lookback period.
        halflife : int, optional
            The halflife for decay.
        bench : str
            The benchmark name.
        portfolio_type : str
            The portfolio type.

        Returns
        -------
        pd.DataFrame
            The calculated beta factor.
        -----------------------------------------------------------------------
        从贝塔相关指标中提取贝塔因子.

        参数
        ----
        periods : int
            回看窗口.
        halflife : int, 可选
            半衰期.
        bench : str
            基准名称.
        portfolio_type : str
            组合类型.

        返回
        ----
        pd.DataFrame
            计算得到的贝塔因子.
        -----------------------------------------------------------------------
        """
        return cls._beta2(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type).beta
