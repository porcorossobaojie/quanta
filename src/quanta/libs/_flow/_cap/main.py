# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:11:56 2026

@author: Porco Rosso
"""
from functools import lru_cache
import numpy as np
import pandas as pd

from quanta.libs import _pandas
from quanta.libs._flow._main import __instance__
from quanta.config import settings
config = settings('flow').cap

class Series(pd.Series):
    _internal_names = pd.Series._internal_names + []
    _internal_names_set = set(_internal_names)
    _metadata = pd.Series._metadata  + [f'_{i}' for i in config.recommand_settings.keys()]

    def __repr__(self) -> str:
        x = super().__repr__()
        x = x + '\nstate: %s, unit: %s, \ncount: %s, cash: %s, \nis_adj: %s' %(self.state, self.unit, len(self), round(self.cash, 3), self.is_adj)
        return x

    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_sliced(self):
        return Series

    @classmethod
    @lru_cache(maxsize=16)
    def __get_values__(cls, portfolio_type, key, name):
        x = __instance__[portfolio_type](key).loc[name]
        return x

    def __pos_neg_rebalance__(self, zero_adj=True, total_weight=None, **kwargs):
        pos = self > 0
        x = self.copy()
        if zero_adj: # 平衡买卖
            x[pos] = x[pos] / x[pos].sum(),
            x[~pos] = x[~pos] / x[~pos].sum()
            if total_weight is not None:
                x = x * total_weight
        else: # 计算多空差异,以差异为权重1,再*total_weight
            w = np.abs(x.sum()) * (total_weight if total_weight is not None else 1)
            x = x / w
        return x

    def ___lazymem_clean__(self):
        attrs = {i for i in self.__dict__.keys() if '__lazymem_' in i}
        [delattr(self, i) for i in attrs]

    def __add__(self, other: pd.Series) -> 'Series':
        if isinstance(other, pd.Series):
            index = self.index.union(other.index)
            self = self.reindex(index, fill_value=0)
            other = other.reindex(index, fill_value=0)
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit isnot match...')
            x = super().__add__(other)
            x.cash = x.cash + getattr(other, 'cash', 0)
            x.name = max(self.name,pd.to_datetime(other.name))
        else:
            x = super().__add__(other)
        return x

    def __radd__(self, other: pd.Series) -> 'Series':
        return self.__add__(other)

    def __sub__(self, other: pd.Series) -> 'Series':
        if isinstance(other, pd.Series):
            index = self.index.union(other.index)
            self = self.reindex(index, fill_value=0)
            other = other.reindex(index, fill_value=0)
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit is not match...')
            x = super().__sub__(other)
            x.cash = x.cash - getattr(other, 'cash', 0)
            x.name = max(self.name,pd.to_datetime(other.name))
        else:
            x = super().__sub__(other)
        return x

    def __rsub__(self, other: pd.Series) -> 'Series':
        return self.__sub__(other)

    def __mul__(self, others: [int, float, np.number, pd.Series]) -> 'Series':
        x = super().__mul__(others)
        x.name = self.name
        return x

    def __truediv__(self, others: [int, float, np.number, pd.Series]) -> 'Series':
        x = super().__truediv__(others)
        x.name = self.name
        return x

    def __init__(self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        fastpath=False,
        **kwargs
    ):
        params = config.recommand_settings.to_dict() | kwargs
        [setattr(self, f'_{i}',j) for i,j in params.items()]
        super().__init__(data, index, dtype, name, copy, fastpath)

    def __zero_check__(self):
        if (self.values == 0).all():
            raise ValueError("<WARNING>: all of series values is 0.")

    def __weight_to_weight__(self, zero_adj=False, total_weight=None, **kwargs):
        self.__zero_check__()
        x = self.__pos_neg_rebalance__(zero_adj, total_weight)
        x._unit = 'weight'
        return x

    def __weight_to_assets__(self, cash, zero_adj=False, total_weight=None, **kwargs):
        self.__zero_check__()
        x = self.__pos_neg_rebalance__(zero_adj, total_weight) * cash
        x._unit = 'assets'
        return x

    def __weight_to_share__(self, cash, zero_adj=False, total_weight=None, **kwargs):
        self.__zero_check__()
        x = self.__pos_neg_rebalance__(zero_adj, total_weight) * cash / self.current()
        x._unit = 'share'
        return

    def __assets_to_weight__(self, zero_adj=False, total_weight=None, **kwargs):
        self.__zero_check__()
        x = self.__pos_neg_rebalance__(zero_adj, total_weight)
        x._unit = 'weight'
        return x

    def __assets_to_assets__(self, **kwargs):
        self.__zero_check__()
        return self.copy()

    def __assets_to_share__(self,**kwargs):
        self.__zero_check__()
        x = self / self.current()
        x._unit = 'share'
        return x

    def __share_to_weight__(self, zero_adj=False, total_weight=None, **kwargs):
        self.__zero_check__()
        x = (self * self.current()).__assets_to_weight__(zero_adj, total_weight)
        x._unit = 'weight'
        return x

    def __share_to_assets__(self, **kwargs):
        self.__zero_check__()
        x = self * self.current()
        self._unit = 'assets'
        return x

    def __sahre_to_share__(self, **kwargs):
        return self.copy()

    @property
    def portfolio_type(self):
        return self.index.name.split('_')[0]

    @property
    def name(self) -> pd.Timestamp:
        return self._name
    @name.setter
    def name(self, trade_dt: pd.Timestamp):
        trade_dt = pd.to_datetime(trade_dt)
        if (getattr(self, '_name', None) is not None
            and (self._unit == 'share')
            and (self._is_adj == False)
        ):
            post_adj = (
                self.__get_values__(
                    self.portfolio_type, config.post_factor, trade_dt
                ) /
                self.__get_values__(
                    self.portfolio_type, config.post_factor, self.name
                )
            ).reindex(self.index)
            self.values[:] = self.values * post_adj
        self.___lazymem_clean__()
        self._name = trade_dt

    @property
    def cash(self):
        if self.state == 'settle':
            return self._cash
        elif self.state in ['signal', 'weight']:
            return 0
        else:
            if self.unit == 'share':
                if not hasattr(self, '__lazymem_cash__'):
                    x  = (self * self.current()).sum() * -1
                    setattr(self, '__lazymem_cash__', x)
                return getattr(self, '__lazymem_cash__')
            else:
                return x.sum() * -1
    @cash.setter
    def cash(self, v):
        self._cash = v

    @property
    def is_adj(self):
        return self._is_adj
    @is_adj.setter
    def is_adj(self, v):
        if (self.unit == 'share') and (self._is_adj != v):
            post_adj =  self.__get_values__(
                self.portfolio_type, config.post_factor, self.name
            ).reindex(self.index)
            self.values[:] = self.values * post_adj
        self.___lazymem_clean__()
        self._is_adj = v

    @property
    def state(self):
        return self._state

    @property
    def unit(self):
        return self._unit

    def current(self, is_adj=None, reindex=True):
        key = self._price_mapping.get(self._state)
        is_adj = self.is_adj if is_adj is None else is_adj
        key = key +'_adj' if is_adj else key
        x = self.__get_values__(self.portfolio_type, key, self.name)
        if reindex:
            x = x.reindex(self.index)
        return x

    def weight(self):
        x = getattr(self, f"__{self.unit}_to_weight__")(copy=True)
        x._unit = 'weight'
        return x

    def assets(self, cash=None):
        x = getattr(self, f"__{self.unit}_to_assets__")(cash=cash, copy=True)
        x._unit = 'assets'
        return x

    def share(self, cash=None):
        x = getattr(self, f"__{self.unit}_to_share__")(cash=cash, copy=True)
        x._unit = 'share'
        return x

    def signal(self, shift=-1):
        if self.state != 'signal':
            x = self.f.shift(shift)
            x._state = 'signal'
        else:
            x = self.copy()
        return x

    def order(self, shift=1):
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.shift(shift)
        x._state = 'order'
        return x

    def trade(self, shift=1):
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.shift(shift)
        x._state = 'trade'
        return x

    def settle(self, shift=1):
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.shift(shift)
        x._state = 'settle'
        return x

    def to(self, unit_or_state, **kwargs):
        return getattr(self, unit_or_state)(**kwargs)

    def entrade(self, auto_state=True, shift=1):
        x = self.copy()
        if auto_state:
            x = x.trade(shift)
        tradestatus = ~x.__get_values__(x.portfolio_type, config.tradestatus, x.name).astype('bool')
        return x[tradestatus]
    def untrade(self, auto_state=True, shift=1):
        x = self.copy()
        if auto_state:
            x = x.trade(shift)
        tradestatus = x.__get_values__(x.portfolio_type, config.tradestatus, x.name).astype('bool')
        return x[tradestatus]

    def enbuy(self, auto_state=True, shift=1):
        x = self[self > 0]
        if auto_state:
            x = x.trade(shift)
        buy = 1 - x.__get_values__(x.portfolio_type, config.trade_price, x.name) / x.__get_values__(x.portfolio_type, config.high_limit, x.name) >= self._untrade_limit
        return x[buy]
    def unbuy(self, auto_state=True, shift=1):
        x = self[self > 0]
        if auto_state:
            x = x.trade(shift)
        buy_limit = 1 - x.__get_values__(x.portfolio_type, config.trade_price, x.name) / x.__get_values__(x.portfolio_type, config.high_limit, x.name) < self._untrade_limit
        return x[buy_limit]

    def ensell(self, auto_state=True, shift=1):
        x = self[self <= 0]
        if auto_state:
            x = x.trade(shift)
        sell = 1 - x.__get_values__(x.portfolio_type, config.low_limit, x.name) / x.__get_values__(x.portfolio_type, config.trade_price, x.name) >= self._untrade_limit
        return x[sell]
    def unsell(self, auto_state=True, shift=1):
        x = self[self <= 0]
        if auto_state:
            x = x.trade(shift)
        sell_limit = 1 - x.__get_values__(x.portfolio_type, config.low_limit, x.name) / x.__get_values__(x.portfolio_type, config.trade_price, x.name) < self._untrade_limit
        return x[sell_limit]

    def trade_cost(self, trade_check=True, **kwargs):
        if trade_check:
            x = self.entrade()
            x = x.enbuy(**kwargs).cash * -1 + self.ensell(**kwargs).cash
            x = x * self._trade_cost
        else:
            x = self.abs().cash * self._trade_cost
            return x












