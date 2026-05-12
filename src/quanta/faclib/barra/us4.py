# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:23:15 2026

@author: Porco Rosso
"""

from functools import lru_cache
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from quanta import flow
from quanta.libs.utils._decorator import doc_inherit
from quanta.faclib.barra._base import main as meta
#from._base import main as meta

class main():
    """
    Implementation of Barra USA4 factor model. | Barra USA4 因子模型实现.
    """
    _model_name = 'us4'
    _base = meta
    finance = _base.finance
    trade = _base.trade
    
    @classmethod
    @doc_inherit(meta.bench)
    def bench(cls, code: str, weight: Optional[Union[str, pd.DataFrame]] = None) -> pd.Series:
        return cls._base.bench(code, weight)
    
    @classmethod
    @doc_inherit(meta.size)
    def size(cls) -> pd.DataFrame:
        return cls._base.size()
    
    @classmethod
    @doc_inherit(meta.non_size)
    def non_size(cls) -> pd.DataFrame:
        return cls._base.non_size()

    @classmethod
    @doc_inherit(meta.bm)
    def bm(cls) -> pd.DataFrame:
        return cls._base.bm()
    
    @classmethod
    @doc_inherit(meta.beta)
    def beta(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._base.beta(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type)

    @classmethod
    @lru_cache(maxsize=4)
    def momentum(
        cls,
        long_periods: int = 504,
        short_periods: int = 21,
        halflife: int = 126,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the momentum factor.

        Parameters
        ----------
        long_periods : int
            The long-term lookback period.
        short_periods : int
            The short-term gap period.
        halflife : int
            The halflife for exponential weighting.
        bench : str
            The benchmark for relative return calculation.
        portfolio_type : str
            The portfolio type.

        Returns
        -------
        pd.DataFrame
            The calculated momentum factor.
        -----------------------------------------------------------------------
        计算动量因子.

        参数
        ----
        long_periods : int
            长期回看窗口.
        short_periods : int
            短期空窗期.
        halflife : int
            指数加权的半衰期.
        bench : str
            计算超额收益使用的基准.
        portfolio_type : str
            组合类型.

        返回
        ----
        pd.DataFrame
            计算得到的动量因子.
        -----------------------------------------------------------------------
        """
        ret = getattr(flow, portfolio_type)(cls.trade.returns).f.tradestatus().tools.log(abs_adj=False).astype('float32')
        bench = cls.bench(bench).tools.log().astype('float32')
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns).f.tradestatus()
        w = pd.tools.halflife(long_periods+short_periods, halflife)[short_periods:]
        
        ret_mom = ret.gen.roll_weight(w)
        bench_mom = bench.gen.roll_weight(w)
        x = (ret_mom - bench_mom).shift(short_periods)
        x = x.f.tradestatus(long_periods, halflife)
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def resid_volatility(
        cls,
        periods: int = 252,
        portfolio_type: str = 'astock',
        **kwargs
    ) -> pd.DataFrame:
        """Calculate residual volatility factor | 计算残差波动率因子"""
        x = (
            0.74 * cls._base.dastd(periods=periods, portfolio_type=portfolio_type) +
            0.16 * cls._base.cmra(periods=periods, portfolio_type=portfolio_type) +
            0.10 * cls._base.hsigma(periods=periods, portfolio_type=portfolio_type, **kwargs)
        )
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def liquidity(cls) -> pd.DataFrame:
        """Calculate liquidity factor | 计算流动性因子"""
        x = pd.concat([cls._base.month_turnover(), cls._base.quarter_turnover(), cls._base.annual_turnover()], axis=1)
        x = x.groupby(x.columns, axis=1).mean()
        return x
    
    @classmethod
    @lru_cache(maxsize=4)
    def earnings(cls) -> pd.DataFrame:
        """Calculate earnings factor | 计算盈利因子"""
        cp = cls._base.cp()
        ep = cls._base.ep()
        exep = cls._base.ex_ep()
        x = cp * 0.21 + ep * 0.11 + exep * 0.68
        return x
 
    @classmethod
    @lru_cache(maxsize=4)
    def growth(cls, periods: int = 20) -> pd.DataFrame:
        """Calculate growth factor | 计算成长因子"""
        df = flow.astock.finance(cls.finance.net_profit, shift=4, periods=periods, quarter_adj=3)
        df = df / df.groupby(cls.trade.trade_dt).transform('mean').abs()
        df = df.unstack(cls.trade.trade_dt).T
        trend = pd.DataFrame(
            np.arange(periods).repeat(df.shape[0]).reshape(periods, -1).T,
            columns = np.arange(periods) - periods + 1,
            index = df.index)
        x = df.stats.neutral(fac=trend, dtype='float32', periods=periods, resid=False) 
        net_profit = x.params.fac.iloc[:, -1].unstack(cls.trade.astock_code)
        
        df = flow.astock.finance(cls.finance.oper_rev, shift=4, periods=periods, quarter_adj=3)
        df = df / df.groupby(cls.trade.trade_dt).transform('mean').abs()
        df = df.unstack(cls.trade.trade_dt).T
        trend = pd.DataFrame(
            np.arange(periods).repeat(df.shape[0]).reshape(periods, -1).T,
            columns = np.arange(periods) - periods + 1,
            index = df.index)
        x = df.stats.neutral(fac=trend, dtype='float32', periods=periods, resid=False) 
        oper_rev = x.params.fac.iloc[:, -1].unstack(cls.trade.astock_code)
        x = net_profit * 0.24 + oper_rev * 0.47
        return x
    
    @classmethod
    @lru_cache(maxsize=4)
    def leverage(cls) -> pd.DataFrame:
        """Calculate leverage factor | 计算杠杆因子"""
        mlev = cls._base.market_leverage()
        dtoa = cls._base.debt_to_asset_ratio()
        blev = cls._base.book_leverage()
        x = (
            mlev * 0.38 + 
            dtoa * 0.35 + 
            blev * 0.27
        )
        return x
    
    @classmethod    
    def neutral(
        cls, 
        df: pd.DataFrame, 
        factors_name: List[str] = ['size', 'non_size', 'beta', 'bm', 'earnings', 'momentum']
    ) -> pd.DataFrame:
        """Neutralize a factor against Barra risk factors | 针对 Barra 风险因子对因子进行中性化"""
        factors = {i:getattr(cls,i)() for i in factors_name}
        x = df.stats.neutral(**factors).resid
        return x
    
        
        
