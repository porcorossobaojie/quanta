# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:40 2026

@author: Porco Rosso
"""
import pandas as pd
from functools import lru_cache
#from .base import Series, DataFrame
from quanta.libs._flow._cap.base import Series, DataFrame

class Unit():
    def __init__(self, signal=None, settle=None, target=None, cash=None, trade_cost=True):
        series_instance = Series._config.recommand_settings.to_dict()
        series_instance.update(
            {k: v for k, v in [('cash', cash), ('trade_cost', trade_cost)] if v is not None}
        )
        params = {'signal':signal, 'settle':settle, 'targret':target}
        params = {i:j for i, j in params.items() if j is not None}
        if not len(params):
            raise ValueError(f"parameters:<{params}> can not be all None...")

        if settle is not None and signal is None:
            self.signal = Series(
                [],
                index = pd.CategoricalIndex([], name=settle.index.name),
                name = settle.name,
                state = 'signal',
                unit = 'share',
                cash = 0
            )
        else:
            self.signal = signal

        if settle is None:
            obj = signal if signal is not None else target
            settle = Series(
                [],
                index = pd.CategoricalIndex([], name=obj.index.name),
                name = obj.name,
                state = 'settle',
                unit = 'share',
                cash = series_instance['cash']
            )
        self.settle = settle
        if target is not None and not isinstance(target, Series):
            target = Series(target, cash=0, unit='weight', state='settle')
            if target.index.name is None:
                target.index.name = self.settle.index.name
            if target.name is None:
                target.name = self.settle.name
        self._target = target
        self._trade_cost = trade_cost

        self._meta_attrs = series_instance | {'signal':signal, 'settle':settle, 'target':target}

    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self, v):
        if isinstance(v, Series):
            v = v.signal().share()
        elif v is None:
            pass
        else:
            v = Series(v, state='signal', unit='share')
        setattr(self, '_signal', v)

    @property
    def order(self):
        return self.signal.order() if self.signal is not None else None

    @property
    def trade(self):
        return self.order.trade() if self.signal is not None else None

    @property
    def entrade(self):
        x = self.trade
        x = (x.enbuy() + x.ensell()) if self.signal is not None else None
        return x

    @property
    def settle(self):
        return self._settle
    @settle.setter
    def settle(self, v):
        if not isinstance(v, Series):
            v = Series(v, state='settle', unit='share')
        self._settle = v

    @property
    def trade_cost(self):
        return self.entrade.trade_cost(trade_check=False) if self.signal is not None else 0

    @lru_cache(maxsize=16)
    def roll(self):
        if self.signal is not None:
            settle = self.settle.share()
            entrade = self.entrade.share()
            settle = settle + entrade
            settle = settle.settle()
            if self._trade_cost:
                settle.cash = settle.cash + self.trade_cost
        else:
            settle = self.settle.f.day_shift(1)
        return settle

    @property
    def target(self):
        if self._target is not None:
            return self._target
        else:
            roll = self.roll()
            target = roll.share(roll.total_assets() - self.trade_cost)
            target.cash = 0
            return target

    def _target_setter(self, v):
        roll = self.roll()
        x = Series(v, state='settle', unit='weight').weight().share(roll.total_assets())
        self._target = x

    def __call__(self, new_target):
        settle = self.roll()
        self._target_setter(new_target)
        signal = self.target - settle
        x = Unit(signal=signal, settle=settle, target=None)
        return x

class Chain:
    def __init__(self, dataframe, cash=10000, trade_cost=True):
        self.cash = cash
        self.trade_cost = trade_cost
        self._obj = dataframe
    def __call__(self):
        dic = {}
        unit_obj = Unit(target=self._obj.iloc[0], cash=self.cash, trade_cost=self.trade_cost)
        dic[unit_obj.settle.name] = unit_obj
        for i,j in self._obj.iloc[1:].iterrows():
            if i.month != unit_obj.settle.name.month:
                print(i)
            dic[i] = unit_obj(j)

        return dic


'''
from quanta import flow
ret = flow.astock('ret')
series = Series(ret.iloc[200, :200]).abs().share(1000)
self = Unit(signal=eries(ret.iloc[200, :200]).abs().share(1000), cash=1000)
new_target = Series(ret.iloc[300, 100:300]).abs()
g1 = self(new_target)
print(1)
'''
