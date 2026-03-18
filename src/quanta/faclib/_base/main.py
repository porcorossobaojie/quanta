# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:32:16 2026

@author: Porco Rosso
"""

from functools import lru_cache
from quanta import flow
from quanta.config import settings


class main():
    index_mapping = settings('flow').index_mapping
    trade = settings('flow').trade_keys
    
    @classmethod
    @lru_cache(maxsize=16)
    def bench(cls, code, weight=None):
        code = cls.index_mapping.get(code, code)
        code = code.split('-')
        if len(code) == 1:
            x = flow.aindex(cls.returns)[code[0]]
        elif len(code) == 2:
            x = flow.astock.multilize(code[0])[int(code[1]) if code[1].isdigit() else code[1]]
            x = flow.astock(cls.returns).reindex(x)[x]
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
    
    
