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

class main(meta):
    """
    Base class for Barra factor calculations. | Barra因子计算基类.
    """

    @classmethod
    def size(cls) -> pd.DataFrame:
        """Calculate the size factor (log market cap) | 计算市值因子 (对数市值)"""
        x = (flow.astock(cls.finance.val_mv) * 1e8).tools.log()
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def bm(cls) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the book-to-market factor, neutralized against size.

        Returns
        -------
        pd.DataFrame
            The residuals of book-to-market ratio against the size factor.
        -----------------------------------------------------------------------
        计算账面市值比因子, 并针对市值进行中性化处理.

        返回
        ----
        pd.DataFrame
            账面市值比相对于市值因子的残差.
        -----------------------------------------------------------------------
        """
        x = flow.astock(cls.finance.pb) ** -1
        x = x.stats.neutral(cls.size()).resid
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def non_size(cls) -> pd.DataFrame:
        """
        =======================================================================
        Calculate the non-linear size factor by taking the residuals of
        size^3 against size.

        Returns
        -------
        pd.DataFrame
            The calculated non-linear size factor.
        -----------------------------------------------------------------------
        通过计算市值三次方相对于市值的残差来计算非线性市值因子.

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
        Vectorized high-performance calculation of beta-related metrics using
        einsum and array rolling.

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
            A MultiIndex DataFrame containing alpha, beta, and resid.
        -----------------------------------------------------------------------
        使用 einsum 和数组滚动进行贝塔相关指标的向量化高性能计算.

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
            包含 alpha, beta 和 resid 的多重索引数据框.
        -----------------------------------------------------------------------
        """
        halflife = periods//4 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.trade.returns).astype('float32').f.tradestatus()
        bench = cls.bench(bench)
        bench = pd.DataFrame(bench.values.repeat(ret.shape[1]).reshape(-1, ret.shape[1]), index=ret.index, columns=ret.columns)
        w = pd.tools.halflife(periods, halflife).astype('float32')
        beta = ret.stats.neutral(beta=bench, neu_axis=0, w=w, periods=periods)
        alpha, beta, resid = (
            beta.params.const,
            beta.params.beta,
            pd.DataFrame(
                beta.resid.std(beta.resid.values),
                index = beta.resid.index,
                columns = beta.resid.columns
            )
        )
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
        return cls._beta(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type).beta

    @classmethod
    @lru_cache(maxsize=4)
    def hsigma(
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
    def dastd(
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
        w = pd.tools.halflife(periods, halflife)
        x = (ret - ret.gen.roll_weight(w)) ** 2
        x = x.gen.roll_weight(w)
        x = x[ret.notnull()]
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def cmra(cls, periods: int = 252, portfolio_type: str = 'astock') -> pd.DataFrame:
        """Calculate Cumulative Range of Adjusted Returns (CMRA) | 计算累积相对收益范围"""
        ret = getattr(flow, portfolio_type)(cls.trade.returns).fillna(0)
        location = -(np.arange(0, periods, 21) + 1)
        x = ret.rolling(periods).sum().tools.log()
        df = x.rolling(periods).apply(lambda x: np.max(x[location], axis=0) - np.min(x[location], axis=0), raw=True)
        df = df.tools.log().f.tradestatus(periods=periods)
        return df

    @classmethod
    @lru_cache(maxsize=4)
    def _turnover(cls, periods, portfolio_type: str = 'astock') -> pd.DataFrame:
        x = getattr(flow, portfolio_type)(cls.finance.turnover) / 100
        x = x.rolling(periods, min_periods=periods // 4).sum()
        x = x.tools.log()
        return x

    @classmethod
    def month_turnover(cls, periods=21, portfolio_type: str = 'astock') -> pd.DataFrame:
        x = cls._turnover(periods=periods, portfolio_type = portfolio_type)
        return x

    @classmethod
    def quarter_turnover(cls, periods=63, portfolio_type: str = 'astock') -> pd.DataFrame:
        x = cls._turnover(periods=periods, portfolio_type = portfolio_type)
        x = x - np.log(4)
        return x

    @classmethod
    def annual_turnover(cls, periods=252, portfolio_type: str = 'astock') -> pd.DataFrame:
        x = cls._turnover(periods=periods, portfolio_type = portfolio_type)
        x = x - np.log(12)
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def annul_weight_turnover(cls, periods=252, portfolio_type: str = 'astock') -> pd.DataFrame:
        df = getattr(flow, portfolio_type)(cls.finance.turnover).f.tradestatus() / 100
        w = pd.tools.halflife(periods, halflife=periods // 4)
        x = df.gen.roll_weight(w)
        x = x[df.notnull()]
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def short_term_reversal(cls, periods=21, portfolio_type: str = 'astock') -> pd.DataFrame:
        w = pd.tools.halflife(periods, periods // 4)
        ret = getattr(flow, portfolio_type)(cls.trade.returns).f.tradestatus().tools.log(abs_adj=False)
        x = ret.gen.roll_weight(w)
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def seasonal(cls, periods=5, window=252, future=21, portfolio_type: str = 'astock') -> pd.DataFrame:
        ret = getattr(flow, portfolio_type)(cls.trade.returns).f.tradestatus()
        ret = ret.rolling(future).mean().shift(window - future + 1)
        df = {i:ret.shift(i * window) for i in range(periods)}
        df = pd.concat(df, axis=1).stack().mean(axis=1).unstack().reindex_like(ret).f.tradestatus()
        return df

    @classmethod
    @lru_cache(maxsize=4)
    def industry_momentum(cls, periods=126, industry_code = 'swl1_name', mv_rebalance=True):
        w = pd.tools.halflife(periods, periods // 6)
        mv = flow.astock(cls.finance.cur_mv) ** 0.5 *1e4
        ret = flow.astock(cls.trade.returns).f.tradestatus()
        x = ret.gen.roll_weight(w)
        x = x * mv
        mv = mv.f.label(industry_code).groupby(industry_code, axis=1).sum(min_count=1).f.expand()
        mv = mv.groupby(cls.trade.astock_code, axis=1).sum(min_count=1)
        i_mom = x.f.label(industry_code)
        i_mom = i_mom.groupby(industry_code, axis=1).sum(min_count=1).f.expand()
        i_mom = i_mom.groupby(cls.trade.astock_code, axis=1).sum(min_count=1)
        mom = (i_mom - x)
        if mv_rebalance:
            mom = mom / mv
        return mom

    @classmethod
    @lru_cache(maxsize=4)
    def relative_strengh(cls, periods = 252, shift=11):
        w = pd.tools.halflife(periods, periods // 2)
        ret = flow.astock(cls.trade.returns).f.tradestatus()
        x = ret.gen.roll_weight(w)
        x = x.rolling(shift).mean().shift(shift)
        return x

    @classmethod
    def historical_alpha(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._beta(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type).alpha

    @classmethod
    @lru_cache(maxsize=1)
    def market_leverage(cls):
        mv = flow.astock(cls.finance.val_mv) * 1e8
        long_liab = flow.astock.finance(cls.finance.long_liab, shift=4, periods=1)
        x = (mv + long_liab) / long_liab
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def book_leverage(cls):
        mv = flow.astock(cls.finance.val_mv) * 1e8
        long_liab = flow.astock.finance(cls.finance.long_liab, shift=4, periods=1)
        pb = flow.astock(cls.finance.val_mv)
        x = pb ** -1 + (long_liab / mv)
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def debt_to_asset_ratio(cls):
        liab = flow.astock.finance(cls.finance.total_liab, periods=1, shift=4)
        asset = flow.astock.finance(cls.finance.total_assets, periods=1, shift=4)
        x = liab / asset
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def _variation(cls, key, periods, quarter_adj):
        df = flow.astock.finance(key, shift=4, periods=periods, quarter_adj=quarter_adj)
        df = df / df.groupby(cls.trade.trade_dt).transform('mean').replace(0, np.nan)
        df = df.groupby(cls.trade.trade_dt).std()
        return df

    @classmethod
    def variation_in_sales(cls, periods=20):
        df = cls._variation(cls.finance.oper_rev, periods, 3)
        return df

    @classmethod
    def variation_to_earnings(cls, periods=20):
        df = cls._variation(cls.finance.net_profit, periods, 3)
        return df

    @classmethod
    def variation_to_cashflow(cls, periods=20):
        df = cls._variation(cls.finance.net_cashflow, periods, None)
        return df

    @classmethod
    @lru_cache(maxsize=1)
    def accr_balancesheet(cls):
        da = pd.concat(
            {i: flow.astock.finance(i, periods=1, shift=8) for i in
                 [cls.finance.deprec, cls.finance.inv_depre, cls.finance.inta_amort, cls.finance.long_amort]
            },
            axis=1
        )
        da = da.groupby(cls.trade.astock_code, axis=1).sum(min_count=1).fillna(0).f.tradestatus()
        ta = flow.astock.finance(cls.finance.total_assets, periods=2, shift=4).swaplevel(-1,0)
        tc = flow.astock.finance(cls.finance.cash_and_equity, periods=2, shift=4).swaplevel(-1,0)
        x = (ta.loc[0] - ta.loc[-1]) - (tc.loc[0] - tc.loc[-1])
        x = (x - da) / ta.loc[0]
        return x

    @classmethod
    @lru_cache(maxsize=1)
    def accr_cashflow(cls):
        da = pd.concat(
            {i: flow.astock.finance(i, periods=1, shift=8) for i in
                 [cls.finance.deprec, cls.finance.inv_depre, cls.finance.inta_amort, cls.finance.long_amort]
            },
            axis=1
        )
        da = da.groupby(cls.trade.astock_code, axis=1).sum(min_count=1).fillna(0).f.tradestatus()
        ni = flow.astock.finance(cls.finance.net_profit, shift=4, periods=1, quarter_adj=3)
        cfo = flow.astock.finance(cls.finance.oper_cash, shift=4, periods=1, quarter_adj=3)
        cfi = flow.astock.finance(cls.finance.inv_cash, shift=4, periods=1, quarter_adj=3)
        ta =flow.astock.finance(cls.finance.total_assets, periods=1, shift=4)
        x = ni - (cfo + cfi) + da
        x = x / ta
        return x













