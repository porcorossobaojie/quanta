# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:36:01 2026

@author: Porco Rosso
"""

import pandas as pd
from typing import Optional, List
from quanta.libs.utils import lru_cache

from quanta import faclib
#char = settings('trade').strategy_001.BJ_13611823855
from quanta.trade import account

class main:
    """
    ===========================================================================
    Base strategy class providing the standard factor-based trading workflow:
    pool construction, ranking, sell/hold/buy decisions, rebalancing, and
    order writing.
    ---------------------------------------------------------------------------
    基础策略类, 提供基于因子的标准交易流程: 股票池构建, 排序, 卖出/持有/买入
    决策, 再平衡及下单写入.
    ---------------------------------------------------------------------------
    """

    @classmethod
    def _internal_data(cls) -> None:
        """Internal data placeholder | 内部数据占位"""
        pass
    
    @classmethod
    def factor(cls) -> pd.DataFrame:
        """Defines the base factor for the strategy | 定义策略的基础因子"""
        return faclib.barra.us4.bm()
    
    def __init__(self, account_obj: 'account') -> None:
        """Initializes the strategy with a trade account | 用交易账户初始化策略"""
        self.account = account_obj
    
    @lru_cache(maxsize=4)
    def pool(self, index_members: Optional[str] = None) -> pd.Series:
        """Builds the tradable pool from the factor with filters | 基于过滤条件从因子构建可交易股票池"""
        x = self.factor().f.filtered()
        if index_members is not None:
            x = x.f.index_members(index_members)
        x = x.iloc[-1].dropna().sort_index()
        x = pd.f.Series(x)
        return x
    
    def ranker(self, lst: Optional[List[str]] = None, index_members: Optional[str] = None) -> pd.Series:
        """Ranks the tradable pool, optionally restricted to a given list | 对可交易股票池排序, 可选限制在给定列表内"""
        x = self.pool(index_members).rank(ascending=False)
        if lst is not None:
            x = x.loc[lst]
        return x
    
    def settle(self) -> pd.Series:
        """Returns the current positive share positions | 返回当前正持仓份额"""
        x = self.account.settle()
        x = x[x > 0]
        return x
    
    def ensell(self, high_limit: bool = True) -> pd.Series:
        """Determines sell candidates beyond the portfolio size | 确定超出组合规模的卖出候选"""
        x = self.settle()
        sells = x[self.ranker(lst = x.index) > (self.account._portfolio_count + self.account._portfolio_range)]
        if high_limit:
            sells = sells[~sells.f.info('tradestatus').astype('bool') & (sells.f.info('close') < sells.f.info('high_limit'))]
        return sells
    
    def hold(self) -> pd.Series:
        """Returns the positions to keep after selling | 返回卖出后继续持有的仓位"""
        x = self.settle()
        x = x[x.index.difference(self.ensell().index)]
        return x
    
    def enbuy(
        self,
        high_limit: bool = True,
        low_limit: bool = True,
        index_member: Optional[str] = None
    ) -> pd.Series:
        """Determines buy candidates ranked outside current holdings | 确定当前持仓之外的买入候选"""
        rank = self.ranker().sort_values()
        if high_limit:
            rank = rank[~rank.f.info('tradestatus').astype('bool') & (rank.f.info('close') < rank.f.info('high_limit'))]
        if low_limit:
            rank = rank[~rank.f.info('tradestatus').astype('bool') & (rank.f.info('close') > rank.f.info('low_limit'))]
        rank = rank[rank.index.difference(self.hold().index)].nsmallest(self.account._portfolio_count - self.hold().shape[0])
        return rank

    @lru_cache(maxsize=4)    
    def rebalance(
        self,
        hold: Optional[pd.Series] = None,
        ensell: Optional[pd.Series] = None,
        enbuy: Optional[pd.Series] = None,
        min_change: float = 0.5,
        extra_cash: float = 0,
        weight: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        =======================================================================
        Computes the target portfolio and generates trade signals.

        Parameters
        ----------
        hold : Optional[pd.Series]
            Positions to keep. Default is None (from self.hold()).
        ensell : Optional[pd.Series]
            Positions to sell. Default is None (from self.ensell()).
        enbuy : Optional[pd.Series]
            Positions to buy. Default is None (from self.enbuy()).
        min_change : float
            Minimum relative change to trigger a trade. Default is 0.5.
        extra_cash : float
            Extra cash added to the portfolio. Default is 0.
        weight : Optional[pd.Series]
            Target weights. Default is None (equal weight).

        Returns
        -------
        pd.DataFrame
            Rebalance result with settle, hope, signal, filter, and ranker.
        -----------------------------------------------------------------------
        计算目标投资组合并生成交易信号.

        参数
        ----
        hold : Optional[pd.Series]
            保留的仓位. 默认为 None (来自 self.hold()).
        ensell : Optional[pd.Series]
            卖出的仓位. 默认为 None (来自 self.ensell()).
        enbuy : Optional[pd.Series]
            买入的仓位. 默认为 None (来自 self.enbuy()).
        min_change : float
            触发交易的最小相对变动比例. 默认为 0.5.
        extra_cash : float
            添加到投资组合的额外现金. 默认为 0.
        weight : Optional[pd.Series]
            目标权重. 默认为 None (等权).

        返回
        ----
        pd.DataFrame
            包含 settle, hope, signal, filter 和 ranker 的再平衡结果.
        -----------------------------------------------------------------------
        """
        hold = self.hold() if hold is None else hold
        enbuy = self.enbuy() if enbuy is None else enbuy
        portfolio_index = hold.index.union(enbuy.index)
        if weight is None:
            portfolio = pd.f.Series(1, pd.CategoricalIndex(portfolio_index), name=hold.name, state='settle')
        else:
            portfolio = pd.f.Series(weight.reindex(portfolio_index), state='settle')
        portfolio = portfolio.share(self.settle().total_assets() + extra_cash).unadj().round(-2)
        df = pd.concat({'settle':self.settle(), 'hope': portfolio}, axis=1)
        df['signal'] = df['hope'].fillna(0) - df['settle'].fillna(0)
        df['filter'] = df['signal'][(df['signal'] / df['settle'].fillna(1)).abs() > min_change]
        df['ranker'] = self.ranker(df.index)
        return df
        
    def signal(self) -> pd.Series:
        """Extracts the trade signal from the rebalance result | 从再平衡结果中提取交易信号"""
        x = self.rebalance()['filter']
        x = pd.f.Series(x, name=self.settle().name, unit='share', is_adj=False)
        return x
            
    def write(self) -> None:
        """Writes the trade signal to the account pipeline | 将交易信号写入账户管道"""
        df = self.signal()
        df.index = [i.split('.')[0] for i in df.index]
        self.account.pipeline.write(
            df = df.to_frame().reset_index(), 
            path = str(self.account.__order_path__/str(df.name.date()))
        )
        
        
        
        
        
        

            
