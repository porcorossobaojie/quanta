# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:23:15 2026

@author: Porco Rosso
"""

from functools import lru_cache
import numpy as np
import pandas as pd

from quanta import flow

from quanta.factors.barra._base import main as meta
#from._base import main as meta

class main(meta):
    _model_name = 'use4'

    @classmethod
    @lru_cache(maxsize=4)
    def momentum(
        cls,
        long_periods = 504,
        short_periods = 21,
        halflife = 126,
        bench='full',
        portfolio_type = 'astock'

    ):
        ret = getattr(flow, portfolio_type)(cls.returns).tools.log().astype('float32')
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
        periods=252,
        halflife=None,
        portfolio_type='astock'
    ):
        halflife = periods // 6 if halflife is None else halflife
        ret = getattr(flow, portfolio_type)(cls.returns).f.tradestatus()
        w = pd.tools.halflife(periods, halflife)[np.newaxis, :]
        entrade = ret.f.tradestatus().notnull()
        x = (ret - ret.rolling(periods, halflife).mean()).fillna(0) ** 2
        x = x.rolling(periods).apply(lambda x: w @ x, raw=True)
        w_roll = entrade.rolling(periods).apply(lambda x: w @ x, raw=True)
        x = (x / w_roll).f.tradestatus()
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def _cmra(
        cls,
        periods=252,
        portfolio_type='astock'
    ):
        ret = getattr(flow, portfolio_type)(cls.returns).fillna(0)
        location = -(np.arange(0, periods, 21) + 1)
        x = ret.rolling(periods).sum().tools.log()
        df = x.rolling(periods).apply(lambda x: np.max(x[location], axis=0) - np.min(x[location], axis=0), raw=True)
        df = df.tools.log().f.tradestatus(periods=periods)
        return df

    @classmethod
    @lru_cache(maxsize=4)
    def _hsigma(
        cls,
        periods=252,
        halflife=None,
        bench = 'full',
        portfolio_type='astock'
    ):
        x = cls._beta(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type).resid
        x = x.stats.neutral(me=cls.size(), beta=cls.beta()).resid
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def resid_volatility(
        cls,
        periods=252,
        portfolio_type='astock',
        **kwargs
    ):
        x = (
            0.74 * cls._dastd(periods=periods, portfolio_type=portfolio_type) +
            0.16 * cls._cmra(periods=periods, portfolio_type=portfolio_type) +
            0.10 * cls._hsigma(periods=periods, portfolio_type=portfolio_type, **kwargs)
        )
        return x

    @classmethod
    @lru_cache(maxsize=4)
    def liquidity(cls, periods=252):
        p1, p2, p3 = periods, periods//4, periods//12
        mv = flow.astock('free_mv')







