# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:40 2026

@author: Porco Rosso
"""

from box import Box
import pandas as pd
from functools import lru_cache
#from .base import Series, DataFrame
from quanta.libs._flow._cap.base import Series, DataFrame

__all__ = ['Series', 'DataFrame', 'Unit', 'Chain']

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

        if signal is None:
            obj = settle if settle is not None else target
            signal = Series(
                [],
                index = pd.CategoricalIndex([], name=obj.index.name),
                name = obj.name,
                state = 'signal',
                unit = 'share',
                cash = 0
            )
        self.signal = signal


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
        else:
            v = Series(v, state='signal', unit='share')
        setattr(self, '_signal', v)

    @property
    @lru_cache(maxsize=1)
    def order(self):
        return self.signal.order() if self.signal is not None else None

    @property
    @lru_cache(maxsize=1)
    def trade(self):
        return self.order.trade() if self.signal is not None else None

    @property
    @lru_cache(maxsize=1)
    def entrade(self):
        x = self.trade
        x = (x.enbuy() + x.ensell()) if self.signal is not None else None
        return x

    @property
    def settle(self):
        return self._settle
    @settle.setter
    def settle(self, v):
        if isinstance(v, Series):
            v = v.settle().share()
        else:
            v = Series(v, state='settle', unit='share')
        self._settle = v

    @property
    @lru_cache(maxsize=1)
    def trade_cost(self):
        return self.entrade.trade_cost() if self.signal is not None else 0

    @lru_cache(maxsize=16)
    def roll(self):
        if self.signal is not None:
            settle = self.settle.share()
            entrade = self.entrade.share()
            settle = settle + entrade
            settle = settle.settle()
            if self._trade_cost:
                settle.cash = settle.cash - self.trade_cost
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

    def set_target(self, v):
        roll = self.roll()
        x = Series(v, state='settle', unit='weight').weight().share(roll.total_assets())
        self._target = x

    def __call__(self, new_target):
        settle = self.roll()
        self.set_target(new_target)
        signal = self.target - settle
        x = Unit(signal=signal, settle=settle, target=None)
        return x

    def turnover(self, actual=True, theory=True, order=True):
        dic = {}
        if actual:
            try:
                x = round(self.entrade.assets().abs().sum() / self.settle.total_assets(), 3)
            except:
                x = 0
            dic['actual'] = x
        if theory:
            try:
                x = round(self.trade.assets().abs().sum() / self.settle.total_assets(), 3)
            except:
                x = 0
            dic['theory'] = x
        if order:
            try:
                x = round(self.order.assets().abs().sum() / self.settle.total_assets(), 3)
            except:
                x = 0
            dic['order'] = x
        return dic

    def returns(self, actual=True, theory=True, order=True):
        dic = {}
        if actual:
            try:
                x = round(self.roll().total_assets() / self.settle.total_assets() - 1, 3)
            except:
                x = 0
            dic['actual'] = x
        if theory:
            try:
                x = round(((self.settle + self.trade).total_assets() - self.trade.trade_cost() if self._trade_cost else 0)/ self.settle.total_assets() - 1, 3)
            except:
                x = 0
            dic['theory'] = x
        if order:
            try:
                x = round(((self.settle + self.order).total_assets() - self.trade.trade_cost() if self._trade_cost else 0)/ self.settle.total_assets() - 1, 3)
            except:
                x = 0
            dic['order'] = x
        return dic

    def different(self, actual=True, theory=True, order=True):
        turnover = self.turnover(actual, theory, order)
        returns = self.returns(actual, theory, order)
        dic = {'turnover':turnover, 'returns':returns}
        x = Box(default_box=False, box_dots=True)
        x.merge_update(dic)
        return pd.DataFrame(x)

class Chain:
    def __init__(self, dataframe, cash=10000, trade_cost=True):
        self.cash = cash
        self.trade_cost = trade_cost
        self._obj = DataFrame(dataframe)

    def __call__(self):
        dic = {}
        unit_obj = Unit(target=self._obj.iloc[0], cash=self.cash, trade_cost=self.trade_cost)
        dic[unit_obj.settle.name] = unit_obj
        for i,j in self._obj.iloc[1:].iterrows():
            if i.month != unit_obj.settle.name.month:
                print(unit_obj.settle.name, round(unit_obj.roll().total_assets(), 4))
            unit_obj = unit_obj(j)
            dic[i] = unit_obj
            self._internal_data = dic
        return dic

    @property
    @lru_cache(maxsize=1)
    def settle(self):
        if not hasattr(self, '_internal_data'):
            self.__call__()
        x = pd.Series({i:j.settle.total_assets() for i,j in self._internal_data.items()}).shift(-1)
        return x

    @property
    def returns(self):
        return self.settle.pct_change()

    @property
    @lru_cache(maxsize=1)
    def turnover(self):
        if not hasattr(self, '_internal_data'):
            self.__call__()
        settle = self.settle.shift()
        trade = pd.Series({i:j.entrade._cash for i,j in self._internal_data.items() if i !=list(self._internal_data.keys())[-1] })
        x = (trade / settle).abs()
        return x

