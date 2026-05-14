# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:57:31 2026

@author: Porco Rosso
"""

from functools import lru_cache
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from quanta import flow
from quanta.libs.utils import dict_to_dataclass, doc_inherit, filter_class_attrs
from quanta.faclib.barra._base import main as meta

class main():
    """
    Implementation of Barra CN6 factor model. | Barra CN6 因子模型实现.
    """
    _model_name = 'cn6'
    _base = meta
    finance = _base.finance
    trade = _base.trade
    _group_info = {
        'residual_volatility': {'dastd': 0.74, 'cmra': 0.16, 'hsigma': 0.10},
        'liquidity': {'month_turnover': 0.25, 'quarter_turnover': 0.25, 'annual_turnover': 0.25, 'annul_weight_turnover': 0.25},
        'leverage': {'market_leverage': 0.38, 'debt_to_asset_ratio': 0.35, 'book_leverage': 0.27}
        }
    
    @classmethod
    @doc_inherit(meta.bench)
    def bench(cls, code: str, weight: Optional[Union[str, pd.DataFrame]] = None) -> pd.Series:
        return cls._base.bench(code, weight)
    
    @classmethod
    def _group(cls, dic):
        df = {i:getattr(cls, i)() * j for i,j in dic.items()}
        df = pd.concat(df, axis=1)
        df = df.groupby(df.columns.get_level_values(1), axis=1).sum(min_count=len(dic))
    
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
    @doc_inherit(meta.hsigma)
    def hsigma(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._base.hsigma(periods=periods, halflife = halflife, bench = bench, portfolio_type = portfolio_type) 
        
    @classmethod
    @doc_inherit(meta.dastd)
    def dastd(
        cls,
        periods: int = 252,
        halflife: int = None,
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._base.dastd(periods=periods, halflife=halflife, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.cmra)
    def cmra(cls, periods: int = 252, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.cmra(periods=periods, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.month_turnover)
    def month_turnover(cls, periods: int = 21, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.month_turnover(periods=periods, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.quarter_turnover)
    def quarter_turnover(cls, periods: int = 63, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.quarter_turnover(periods=periods, portfolio_type=portfolio_type)
        
    @classmethod
    @doc_inherit(meta.annual_turnover)
    def annual_turnover(cls, periods: int = 252, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.annual_turnover(periods=periods, portfolio_type=portfolio_type)
        
    @classmethod
    @doc_inherit(meta.annul_weight_turnover)
    def annul_weight_turnover(cls, periods: int = 252, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.annul_weight_turnover(periods=periods, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.short_term_reversal)
    def short_term_reversal(cls, periods: int = 21, portfolio_type: str = 'astock') -> pd.DataFrame:
        return cls._base.short_term_reversal(periods=periods, portfolio_type=portfolio_type)

    @classmethod
    @doc_inherit(meta.seasonal)
    def seasonal(
        cls,
        periods: int = 5,
        window: int = 252,
        future: int = 21,
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._base.seasonal(periods=periods, window=window, future=future, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.industry_momentum)
    def industry_momentum(
        cls,
        periods: int = 126,
        industry_code: str = 'swl1_name',
        mv_rebalance: bool = True
    ) -> pd.DataFrame:
        return cls._base.industry_momentum(periods=periods, industry_code=industry_code, mv_rebalance=mv_rebalance)
    
    @classmethod
    @doc_inherit(meta.relative_strengh)
    def relative_strengh(cls, periods: int = 252, shift: int = 11) -> pd.DataFrame:
        return cls._base.relative_strengh(periods=periods, shift=shift)
    
    @classmethod
    @doc_inherit(meta.historical_alpha)
    def historical_alpha(
        cls,
        periods: int = 252,
        halflife: int = None,
        bench: str = 'full',
        portfolio_type: str = 'astock'
    ) -> pd.DataFrame:
        return cls._base.historical_alpha(periods=periods, halflife=halflife, bench=bench, portfolio_type=portfolio_type)
    
    @classmethod
    @doc_inherit(meta.market_leverage)
    def market_leverage(cls) -> pd.DataFrame:
        return cls._base.market_leverage()
    
    @classmethod
    @doc_inherit(meta.market_leverage)
    def book_leverage(cls) -> pd.DataFrame:
        return cls._base.book_leverage()
    
    @classmethod
    @doc_inherit(meta.debt_to_asset_ratio)
    def debt_to_asset_ratio(cls) -> pd.DataFrame:
        return cls._base.debt_to_asset_ratio()
    
    @classmethod
    @doc_inherit(meta.variation_in_sales)
    def variation_in_sales(cls, periods: int = 20) -> pd.DataFrame:
        return cls._base.variation_in_sales(periods=periods)
    
    @classmethod
    @doc_inherit(meta.variation_in_earnings)
    def variation_in_earnings(cls, periods: int = 20) -> pd.DataFrame:
        return cls._base.variation_in_earnings(periods=periods)

    @classmethod
    @doc_inherit(meta.variation_in_cashflow)
    def variation_in_cashflow(cls, periods: int = 20) -> pd.DataFrame:
        return cls._base.variation_in_cashflow(periods=periods)

    @classmethod
    @doc_inherit(meta.accr_balancesheet)
    def accr_balancesheet(cls) -> pd.DataFrame:
        return cls._base.accr_balancesheet()

    @classmethod
    @doc_inherit(meta.accr_cashflow)
    def accr_cashflow(cls) -> pd.DataFrame:
        return cls._base.accr_cashflow()

    @classmethod
    @doc_inherit(meta.asset_turnover)
    def asset_turnover(cls) -> pd.DataFrame:
        return cls._base.asset_turnover()
    
    @classmethod
    @doc_inherit(meta.gross_profit)
    def gross_profit(cls) -> pd.DataFrame:
        return cls._base.gross_profit()
    
    @classmethod
    @doc_inherit(meta.gross_profit_margin)
    def gross_profit_margin(cls) -> pd.DataFrame:
        return cls._base.gross_profit_margin()
    
    @classmethod
    @doc_inherit(meta.roa)
    def roa(cls) -> pd.DataFrame:
        return cls._base.roa()
    
    @classmethod
    @doc_inherit(meta.asset_growth)
    def asset_growth(cls, periods: int = 20) -> pd.DataFrame:
        return cls._base.asset_growth(periods=periods)

    @classmethod
    @doc_inherit(meta.invest_growth)
    def invest_growth(cls, periods: int = 20) -> pd.DataFrame:
        return cls._base.invest_growth(periods=periods)

    @classmethod
    @doc_inherit(meta.ep)
    def ep(cls) -> pd.DataFrame:
        return cls._base.ep()
    
    @classmethod
    @doc_inherit(meta.cp)
    def cp(cls) -> pd.DataFrame:
        return cls._base.cp()
    
    @classmethod
    @doc_inherit(meta.ex_ep)
    def ex_ep(cls) -> pd.DataFrame:
        return cls._base.ex_ep()
    
    @classmethod
    @doc_inherit(meta.enterprise)
    def enterprise(cls) -> pd.DataFrame:
        return cls._base.enterprise()
    
    @classmethod
    @doc_inherit(meta.long_relative_strengh)
    def long_relative_strengh(cls, periods: int = 252 * 4, shift: int = 11) -> pd.DataFrame:
        return cls._base.long_relative_strengh(periods=periods, shift=shift)
    
    @classmethod
    @doc_inherit(meta.long_historical_alpha)
    def long_historical_alpha(cls, periods: int = 252 * 4, shift: int = 11) -> pd.DataFrame:
        return cls._base.long_historical_alpha(periods=periods, shift=shift)
    
    @classmethod
    @doc_inherit(meta.ep_growth)
    def ep_growth(cls, periods: int = 252 * 4) -> pd.DataFrame:
        return cls._base.ep_growth(periods=periods)
    
    @classmethod
    @doc_inherit(meta.op_growth)
    def op_growth(cls, periods: int = 252 * 4) -> pd.DataFrame:
        return cls._base.op_growth(periods=periods)
    
    @classmethod
    @doc_inherit(meta.dp)
    def dp(cls) -> pd.DataFrame:
        return cls._base.dp()

    @classmethod
    def summary(cls):
        def residual_volatility():
            return cls._group(cls._group_info['residual_volatility'])
        
        def liquidity():
            return cls._group(cls._group_info['liquidity'])
        
        def leverage():
            return cls._group(cls._group_info['leverage'])
        
        dic = {
            'size': {
                'size': {
                    'size': cls.size},
                'non_size': {
                    'non_size': cls.non_size}
            },
            'volatility': {
                'beta': {
                    'beta': cls.beta},
                'residual_volatility': {
                    'hsigma': cls.hsigma,
                    'dastd': cls.dastd,
                    'cmra': cls.cmra,
                    '_group': residual_volatility}
            },
            'liquidity': {
                'liquidity': {
                    'month_turnover': cls.month_turnover,
                    'quarter_turnover': cls.quarter_turnover,
                    'annual_turnover': cls.annual_turnover,
                    'annul_weight_turnover': cls.annul_weight_turnover,
                    '_group': liquidity}
            },
            'momentum': {
                'short_term_reversal': {
                    'short_term_reversal': cls.short_term_reversal},
                'seasonal': {
                    'seasonal': cls.seasonal},
                'industry_momentum': {
                    'industry_momentum': cls.industry_momentum},
                'momentum': {
                    'relative_strengh': cls.relative_strengh,
                    'historical_alpha': cls.historical_alpha}
            },
            'quality': {
                'leverage': {
                    'market_leverage': cls.market_leverage,
                    'book_leverage': cls.book_leverage,
                    'debt_to_asset_ratio': cls.debt_to_asset_ratio,
                    '_group': leverage},
                'variation': {
                    'variation_in_sales': cls.variation_in_sales,
                    'variation_in_cashflow': cls.variation_in_cashflow,
                    'variation_in_earnings': cls.variation_in_earnings},    
                'earn_quality': {
                    'accr_balancesheet': cls.accr_balancesheet,
                    'accr_cashflow': cls.accr_cashflow},
                'profitability': {
                    'asset_turnover': cls.asset_turnover,
                    'gross_profit': cls.gross_profit,
                    'gross_profit_margin': cls.gross_profit_margin,
                    'roa': cls.roa},
                'invest_quality': {
                    'asset_growth': cls.asset_growth,
                    'invest_growth': cls.invest_growth},
            },
            'value': {
                'bm': {
                    'bm': cls.bm},
                'earn_yield': {
                    'ep': cls.ep,
                    'cp': cls.cp,
                    'enterprise': cls.enterprise},
                'long_reversal': {
                    'long_relative_strengh': cls.long_relative_strengh,
                    'long_historical_alpha': cls.long_historical_alpha},
            },
            'grwoth': {
                'growth': {
                    'historical_alpha': cls.historical_alpha,
                    'ep_growth': cls.ep_growth}
            },
            'dividend': {
                'dividend':{
                    'dp': cls.dp}
            }
        }
        return dict_to_dataclass(dic, cls._model_name)
            
    @classmethod    
    def neutral(
        cls, 
        df: pd.DataFrame, 
        factors_name: List[str] = ['size', 'non_size', 'beta', 'bm']
    ) -> pd.DataFrame:
        """Neutralize a factor against Barra risk factors | 针对 Barra 风险因子对因子进行中性化"""
        factors = {i:getattr(cls,i)() for i in factors_name}
        x = df.stats.neutral(**factors).resid
        return x         
            
    @classmethod
    def merge(cls, key):
        summary = cls.summary()
        keys = key.split('.')
        x = filter_class_attrs(getattr(getattr(summary, keys[0]), keys[1]))
        df = {i:j() for i,j in x.items() if i != '_group'}
        df = pd.f.merge(*df.values())
        return df
            
            
            












    
