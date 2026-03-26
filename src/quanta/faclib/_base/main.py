# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:32:16 2026

@author: Porco Rosso
"""

import pandas as pd
from typing import Optional, Union
from functools import lru_cache
from quanta import flow
from quanta.config import settings


class main():
    """
    ===========================================================================
    Base class for factor calculation, providing common utilities like
    benchmark return retrieval and configuration mapping.
    ---------------------------------------------------------------------------
    因子计算的基类, 提供通用的实用工具, 如基准收益获取和配置映射.
    ---------------------------------------------------------------------------
    """
    index_mapping = settings('flow').index_mapping
    trade = settings('flow').trade_keys
    
    @classmethod
    @lru_cache(maxsize=16)
    def bench(
        cls,
        code: str,
        weight: Optional[Union[str, pd.DataFrame]] = None
    ) -> pd.Series:
        """
        =======================================================================
        Retrieves benchmark returns based on a specified code, supporting
        index returns, weighted stock portfolios, or custom data sources.

        Parameters
        ----------
        code : str
            The benchmark code or a structured string (e.g., 'source-table-key').
        weight : Optional[Union[str, pd.DataFrame]]
            The weights to apply if the benchmark is a portfolio.
            Default is None.

        Returns
        -------
        pd.Series
            The calculated benchmark periodic returns.
        -----------------------------------------------------------------------
        根据指定代码检索基准收益率, 支持指数收益, 加权股票组合或自定义数据源.

        参数
        ----
        code : str
            基准代码或结构化字符串 (例如 'source-table-key').
        weight : Optional[Union[str, pd.DataFrame]]
            如果基准是组合, 则应用的权重. 默认为 None.

        返回
        ----
        pd.Series
            计算得到的基准周期收益率.
        -----------------------------------------------------------------------
        """
        code = cls.index_mapping.get(code, code)
        code = code.split('-')
        if len(code) == 1:
            x = flow.aindex(cls.trade.returns)[code[0]]
        elif len(code) == 2:
            x = flow.astock.multilize(code[0])[int(code[1]) if code[1].isdigit() else code[1]]
            x = flow.astock(cls.trade.returns).reindex(x)[x]
            if weight is not None:
                weight = flow.astock(weight).reindex_like(x)[x.notnull()]
                x = (x * weight ** 0.5).sum(axis=1, min_count=1) / weight.sum(axis=1, min_count=1)
            else:
                x = x.mean(axis=1)
        else:
            source = getattr(flow, code[0])
            x = source.multilize(code[1])[int(code[2]) if code[2].isdigit() else code[2]]
            x = source(cls.trade.returns).reindex_like(x)[x]
            if weight is not None:
                try:
                    weight = flow.astock(weight).reindex(x)[x.notnull()]
                except:
                    weight = flow.afund(weight).reindex(x)[x.notnull()]
                x = (x * weight ** 0.5).sum(axis=1, min_count=1) / weight.sum(axis=1, min_count=1)
            else:
                x = x.mean(axis=1)
        x.name = code[-1]
        return x
    
    
