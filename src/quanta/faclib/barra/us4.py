# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:23:15 2026

@author: Porco Rosso
"""

from functools import lru_cache
import numpy as np
import pandas as pd

from quanta import flow

#from quanta.factors.barra._base import main as meta
from._base import main as meta

class main(meta):
    """
    Implementation of Barra USA4 factor model. | Barra USA4 因子模型实现.
    """
    _model_name = 'us4'

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
        ret = getattr(flow, portfolio_type)(cls.trade.returns).tools.log().astype('float32')
        entrade = ret.f.tradestatus().notnull()
        bench = cls.bench(bench).tools.log().astype('float32')
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)[entrade].fillna(0)
        w = pd.tools.halflife(long_periods+short_periods, halflife)[np.newaxis, :]

        ret_mom = ret.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)
        w_mom = entrade.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)
        bench_mom = bench.rolling(long_periods).apply(lambda x: w[np.newaxis, :] @ x, raw=True)

        x = ((ret_mom - bench_mom) / w_mom).shift(short_periods)
        x = x.f.tradestatus(long_periods, halflife)
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def _dastd(
        cls,
        periods: int = 252,
        halflife: int = None,
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the Daily Standard Deviation (DASTD).

        Parameters
        ----------
        periods : int
            The lookback period.
        halflife : int, optional
            The halflife for decay.
        portfolio_type : str
            The portfolio type.

        Returns
        -------
        pd.DataFrame
            The calculated DASTD.
        -----------------------------------------------------------------------
        计算日收益率标准差 (DASTD).

        参数
        ----
        periods : int
            回看窗口.
        halflife : int, 可选
            半衰期.
        portfolio_type : str
            组合类型.

        返回
        ----
        pd.DataFrame
            计算得到的 DASTD.
        -----------------------------------------------------------------------
        """
        halflife = periods // 6 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.trade.returns).f.tradestatus()
        w = pd.tools.halflife(periods, halflife)[np.newaxis, :]
        entrade = ret.f.tradestatus().notnull()
        x = (ret - ret.rolling(periods, halflife).mean()).fillna(0) ** 2
        x = x.rolling(periods).apply(lambda x: w @ x, raw=True)
        w_roll = entrade.rolling(periods).apply(lambda x: w @ x, raw=True)
        x = (x / w_roll).f.tradestatus()
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def _cmra(cls, periods: int = 252, portfolio_type: str = 'astock') -> pd.DataFrame:
        """Calculate Cumulative Range of Adjusted Returns (CMRA) | 计算累积相对收益范围"""
        ret = getattr(flow, portfolio_type)(cls.trade.returns).fillna(0)
        location = -(np.arange(0, periods, 21) + 1)
        x = ret.rolling(periods).sum().tools.log()
        df = x.rolling(periods).apply(lambda x: np.max(x[location], axis=0) - np.min(x[location], axis=0), raw=True)
        df = df.tools.log().f.tradestatus(periods=periods)
        return df

    @classmethod
    @lru_cache(maxsize=4)
    def _hsigma(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        """Calculate Historical Sigma | 计算历史残差波动率"""
        x = cls._beta(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type).resid
        x = x.stats.neutral(me=cls.size(), beta=cls.beta()).resid
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
            0.74 * cls._dastd(periods=periods, portfolio_type=portfolio_type) +
            0.16 * cls._cmra(periods=periods, portfolio_type=portfolio_type) +
            0.10 * cls._hsigma(periods=periods, portfolio_type=portfolio_type, **kwargs)
        )
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def liquidity(cls, periods: int = 252) -> pd.DataFrame:
        """Calculate liquidity factor | 计算流动性因子"""
        periods = [periods, periods//4, periods//12]
        turnover = flow.astock(cls.finance.free_turnover)
        x = {i:turnover.rolling(i).sum().tools.log() for i in periods}
        x = pd.concat(x, axis=1).groupby(turnover.columns.name, axis=1).sum(min_count=3) / len(periods)
        x = x.f.tradestatus(periods[0])
        return x
    
    @classmethod
    @lru_cache(maxsize=4)
    def earnings(cls) -> pd.DataFrame:
        """Calculate earnings factor | 计算盈利因子"""
        cp = flow.astock(cls.finance.pcf) ** -1
        ep = flow.astock(cls.finance.pe) ** -1
        x = cp * 0.21 + ep * 0.68
        return x
 
    @classmethod
    @lru_cache(maxsize=4)
    def growth(cls, periods: int = 20) -> pd.DataFrame:
        """Calculate growth factor | 计算成长因子"""
        net_profit = flow.astock.finance(cls.finance.net_profit, shift=2, periods=periods, quarter_adj=3, min_periods=periods//2)
        net_profit = net_profit.stack().unstack(net_profit.index.names[1])
        w = np.arange(periods)
        x = np.polyfit(w, net_profit.fillna(0).values.T, 1)
        x = x[1] / net_profit.mean(axis=1).abs().values
        net_profit = pd.Series(x, index=net_profit.index).unstack() / 4
        net_profit = net_profit.tools.log()
        
        oper_rev = flow.astock.finance(cls.finance.oper_rev, shift=2, periods=periods,  quarter_adj=3, min_periods=periods//2)
        oper_rev = oper_rev.stack().unstack(oper_rev.index.names[1])
        x = np.polyfit(w, oper_rev.fillna(0).values.T, 1)
        x = x[1] / oper_rev.mean(axis=1).abs().values
        oper_rev = pd.Series(x, index=oper_rev.index).unstack() / 4
        
        x = net_profit.stats.standard() * 0.24 + oper_rev.stats.standard() * 0.47
        return x
    
    @classmethod
    @lru_cache(maxsize=4)
    def leverage(cls) -> pd.DataFrame:
        """Calculate leverage factor | 计算杠杆因子"""
        mv = flow.astock(cls.finance.val_mv) * 1e8
        long_liab = flow.astock.finance(cls.finance.long_liab, periods=1, shift=2)
        total_liab = flow.astock.finance(cls.finance.total_liab, periods=1, shift=2)
        total_assets = flow.astock.finance(cls.finance.total_assets, periods=1, shift=2)
        net_assets = total_assets - total_liab
        
        mlev = long_liab / mv + 1
        dtoa = total_liab / total_assets
        blev = long_liab / (net_assets[net_assets > 1e7])
        x = (
            mlev * 0.38 + 
            dtoa * 0.35 + 
            blev * 0.27
        )
        return x
    
    
        
        
