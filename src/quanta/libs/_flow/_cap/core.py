# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:40 2026

@author: Porco Rosso
"""

from box import Box
import pandas as pd
from functools import lru_cache
from typing import Optional, Union, Dict, Any
from quanta.libs._flow._cap.base import Series, DataFrame

__all__ = ['Series', 'DataFrame', 'Unit', 'Chain']

class Unit:
    """
    ===========================================================================
    A single transaction and settlement unit for a specific trade date.

    Attributes
    ----------
    settle : Series
        The settled position at the beginning of the period.
    signal : Series
        The target trade signal for the period.
    _target : Series
        The target portfolio weights.
    _trade_cost : bool
        Whether to account for trading costs.
    ---------------------------------------------------------------------------
    特定交易日的单个交易和结算单元.

    属性
    ----
    settle : Series
        期初已结算的持仓.
    signal : Series
        本期的目标交易信号.
    _target : Series
        目标投资组合权重.
    _trade_cost : bool
        是否计算交易成本.
    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        signal: Optional[Series] = None,
        settle: Optional[Series] = None,
        target: Optional[Union[Series, pd.Series]] = None,
        cash: Optional[float] = None,
        trade_cost: bool = True
    ):
        """Initializes the transaction unit | 初始化交易单元"""
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
    def signal(self) -> Series:
        """Returns the signal position | 返回信号持仓"""
        return self._signal

    @signal.setter
    def signal(self, v: Union[Series, pd.Series]) -> None:
        """Sets the signal position | 设置信号持仓"""
        if isinstance(v, Series):
            v = v.signal().share()
        else:
            v = Series(v, state='signal', unit='share')
        setattr(self, '_signal', v)

    @property
    @lru_cache(maxsize=1)
    def order(self) -> Optional[Series]:
        """Returns the order series | 返回订单序列"""
        return self.signal.order() if self.signal is not None else None

    @property
    @lru_cache(maxsize=1)
    def trade(self) -> Optional[Series]:
        """Returns the trade series | 返回交易序列"""
        return self.order.trade() if self.signal is not None else None

    @property
    @lru_cache(maxsize=1)
    def entrade(self) -> Optional[Series]:
        """Returns tradable instruments | 返回可交易标的"""
        x = self.trade
        x = (x.enbuy() + x.ensell()) if self.signal is not None else None
        return x

    @property
    def settle(self) -> Series:
        """Returns the settled position | 返回已结算持仓"""
        return self._settle

    @settle.setter
    def settle(self, v: Union[Series, pd.Series]) -> None:
        """Sets the settled position | 设置已结算持仓"""
        if isinstance(v, Series):
            v = v.settle().share()
        else:
            v = Series(v, state='settle', unit='share')
        self._settle = v

    @property
    @lru_cache(maxsize=1)
    def trade_cost(self) -> float:
        """Calculates trading costs | 计算交易成本"""
        return self.entrade.trade_cost() if self.signal is not None else 0

    @lru_cache(maxsize=16)
    def roll(self) -> Series:
        """Rolls position to the next period | 将持仓滚动至下一期"""
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
    def target(self) -> Series:
        """Returns target weights | 返回目标权重"""
        if self._target is not None:
            return self._target
        else:
            roll = self.roll()
            target = roll.share(roll.total_assets() - self.trade_cost)
            target.cash = 0
            return target

    def set_target(self, v: Union[Series, pd.Series]) -> None:
        """Sets the target weights | 设置目标权重"""
        roll = self.roll()
        x = Series(v, state='settle', unit='weight').weight().share(roll.total_assets())
        self._target = x

    def __call__(self, new_target: Union[Series, pd.Series]) -> 'Unit':
        """Executes a transition to a new target | 执行向新目标的转换"""
        settle = self.roll()
        self.set_target(new_target)
        signal = self.target - settle
        x = Unit(signal=signal, settle=settle, target=None)
        return x

    def turnover(
        self,
        actual: bool = True,
        theory: bool = True,
        order: bool = True
    ) -> Dict[str, float]:
        """Calculates portfolio turnover | 计算投资组合换手率"""
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

    def returns(
        self,
        actual: bool = True,
        theory: bool = True,
        order: bool = True
    ) -> Dict[str, float]:
        """Calculates portfolio returns | 计算投资组合收益率"""
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

    def different(
        self,
        actual: bool = True,
        theory: bool = True,
        order: bool = True
    ) -> pd.DataFrame:
        """Generates performance comparison table | 生成绩效对比表"""
        turnover = self.turnover(actual, theory, order)
        returns = self.returns(actual, theory, order)
        dic = {'turnover':turnover, 'returns':returns}
        x = Box(default_box=False, box_dots=True)
        x.merge_update(dic)
        return pd.DataFrame(x)

class Chain:
    """
    ===========================================================================
    A sequence of linked transaction units over time.

    Attributes
    ----------
    cash : float
        Initial cash for the chain.
    trade_cost : bool
        Whether to account for trading costs.
    _obj : DataFrame
        Source target weights for each period.
    ---------------------------------------------------------------------------
    一系列随时间关联的交易单元.

    属性
    ----
    cash : float
        链条的初始现金.
    trade_cost : bool
        是否计算交易成本.
    _obj : DataFrame
        各时期的原始目标权重.
    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cash: float = 10000,
        trade_cost: bool = True
    ):
        """Initializes the backtest chain | 初始化回测链条"""
        self.cash = cash
        self.trade_cost = trade_cost
        self._obj = DataFrame(dataframe)

    def __call__(self) -> Dict[pd.Timestamp, Unit]:
        """Executes the complete backtest chain | 执行完整的向后回测链条"""
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
    def settle(self) -> pd.Series:
        """Returns the settled equity curve | 返回结算后的权益曲线"""
        if not hasattr(self, '_internal_data'):
            self.__call__()
        x = pd.Series({i:j.settle.total_assets() for i,j in self._internal_data.items()}).shift(-1)
        return x

    @property
    def returns(self) -> pd.Series:
        """Returns periodic returns | 返回定期收益率"""
        return self.settle.pct_change()

    @property
    @lru_cache(maxsize=1)
    def turnover(self) -> pd.Series:
        """Calculates turnover curve | 计算换手率曲线"""
        if not hasattr(self, '_internal_data'):
            self.__call__()
        settle = self.settle.shift()
        trade = pd.Series({i:j.entrade._cash for i,j in self._internal_data.items() if i !=list(self._internal_data.keys())[-1] })
        x = (trade / settle).abs()
        return x
