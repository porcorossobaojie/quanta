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
        """
        =======================================================================
        Initializes a transaction and settlement unit.

        Parameters
        ----------
        signal : Optional[Series]
            The target trade signal for the period.
        settle : Optional[Series]
            The settled position at the beginning of the period.
        target : Optional[Union[Series, pd.Series]]
            The target portfolio weights.
        cash : Optional[float]
            Initial cash if settle is None.
        trade_cost : bool
            Whether to account for trading costs. Default is True.
        -----------------------------------------------------------------------
        初始化交易和结算单元.

        参数
        ----
        signal : Optional[Series]
            本期的目标交易信号.
        settle : Optional[Series]
            期初已结算的持仓.
        target : Optional[Union[Series, pd.Series]]
            目标投资组合权重.
        cash : Optional[float]
            如果 settle 为 None, 则为初始现金.
        trade_cost : bool
            是否计算交易成本. 默认为 True.
        -----------------------------------------------------------------------
        """
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
        """
        =======================================================================
        Rolls the current position to the next period based on signals.

        Returns
        -------
        Series
            The settled position for the next period.
        -----------------------------------------------------------------------
        根据信号将当前持仓滚动至下一期.

        返回
        ----
        Series
            下一期的已结算持仓.
        -----------------------------------------------------------------------
        """
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
        x = Series(v.astype(float), state='settle', unit='weight').weight().share(roll.total_assets())
        self._target = x

    def __call__(self, new_target: Union[Series, pd.Series]) -> 'Unit':
        """
        =======================================================================
        Transitions the current unit to a new target state.

        Parameters
        ----------
        new_target : Union[Series, pd.Series]
            The new target weights for the next unit.

        Returns
        -------
        Unit
            A new transaction unit for the next period.
        -----------------------------------------------------------------------
        将当前单元转换到新的目标状态.

        参数
        ----
        new_target : Union[Series, pd.Series]
            下一单元的新目标权重.

        返回
        ----
        Unit
            下一周期的新交易单元.
        -----------------------------------------------------------------------
        """
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
        """
        =======================================================================
        Calculates portfolio turnover metrics.

        Parameters
        ----------
        actual : bool
            Calculate actual turnover (considering trade restrictions).
        theory : bool
            Calculate theoretical turnover (signal-based).
        order : bool
            Calculate order-based turnover.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the requested turnover metrics.
        -----------------------------------------------------------------------
        计算投资组合换手率指标.

        参数
        ----
        actual : bool
            计算实际换手率 (考虑交易限制).
        theory : bool
            计算理论换手率 (基于信号).
        order : bool
            计算基于订单的换手率.

        返回
        ----
        Dict[str, float]
            包含所请求换手率指标的字典.
        -----------------------------------------------------------------------
        """
        dic = {}
        if actual:
            try:
                x = round(self.entrade.assets().abs().sum() / self.settle.total_assets(), 5)
            except:
                x = 0
            dic['actual'] = x
        if theory:
            try:
                x = round(self.trade.assets().abs().sum() / self.settle.total_assets(), 5)
            except:
                x = 0
            dic['theory'] = x
        if order:
            try:
                x = round(self.order.assets().abs().sum() / self.settle.total_assets(), 5)
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
        """
        =======================================================================
        Calculates portfolio return metrics.

        Parameters
        ----------
        actual : bool
            Calculate actual returns.
        theory : bool
            Calculate theoretical returns.
        order : bool
            Calculate order returns.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the requested return metrics.
        -----------------------------------------------------------------------
        计算投资组合收益率指标.

        参数
        ----
        actual : bool
            计算实际收益率.
        theory : bool
            计算理论收益率.
        order : bool
            计算订单收益率.

        返回
        ----
        Dict[str, float]
            包含所请求收益率指标的字典.
        -----------------------------------------------------------------------
        """
        dic = {}
        if actual:
            try:
                x = round(self.roll().total_assets() / self.settle.total_assets() - 1, 5)
            except:
                x = 0
            dic['actual'] = x
        if theory:
            try:
                x = round(((self.settle + self.trade).total_assets() - self.trade.trade_cost() if self._trade_cost else 0)/ self.settle.total_assets() - 1, 5)
            except:
                x = 0
            dic['theory'] = x
        if order:
            try:
                x = round(((self.settle + self.order).total_assets() - self.trade.trade_cost() if self._trade_cost else 0)/ self.settle.total_assets() - 1, 5)
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
        """
        =======================================================================
        Generates a summary comparison table for returns and turnover.

        Parameters
        ----------
        actual : bool
            Include actual metrics.
        theory : bool
            Include theoretical metrics.
        order : bool
            Include order metrics.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the performance comparison.
        -----------------------------------------------------------------------
        生成收益率和换手率的汇总对比表.

        参数
        ----
        actual : bool
            包含实际指标.
        theory : bool
            包含理论指标.
        order : bool
            包含订单指标.

        返回
        ----
        pd.DataFrame
            包含绩效对比的 DataFrame.
        -----------------------------------------------------------------------
        """
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
        """
        =======================================================================
        Initializes the backtest chain.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing target weights over time.
        cash : float
            Initial starting cash. Default is 10000.
        trade_cost : bool
            Whether to enable trading costs. Default is True.
        -----------------------------------------------------------------------
        初始化回测链条.

        参数
        ----
        dataframe : pd.DataFrame
            包含随时间变化的目标权重的 DataFrame.
        cash : float
            初始启动现金. 默认为 10000.
        trade_cost : bool
            是否启用交易成本. 默认为 True.
        -----------------------------------------------------------------------
        """
        self.cash = cash
        self.trade_cost = trade_cost
        self._obj = DataFrame(dataframe)

    def __call__(self) -> Dict[pd.Timestamp, Unit]:
        """
        =======================================================================
        Executes the iterative backtest across all periods in the chain.

        Returns
        -------
        Dict[pd.Timestamp, Unit]
            A dictionary mapping timestamps to the corresponding Unit instances.
        -----------------------------------------------------------------------
        执行链条中所有周期的迭代回测.

        返回
        ----
        Dict[pd.Timestamp, Unit]
            一个将时间戳映射到相应 Unit 实例的字典.
        -----------------------------------------------------------------------
        """
        dic = {}
        unit_obj = Unit(target=self._obj.iloc[0], cash=self.cash, trade_cost=self.trade_cost)
        dic[unit_obj.settle.name] = unit_obj
        for i,j in self._obj.iloc[1:].iterrows():
            if i.month != unit_obj.settle.name.month:
                print(unit_obj.settle.name, round(unit_obj.settle.total_assets(), 4))
            unit_obj = unit_obj(j)
            dic[unit_obj.settle.name] = unit_obj
            self._internal_data = dic
        print(unit_obj.settle.name, round(unit_obj.settle.total_assets(), 4))
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
