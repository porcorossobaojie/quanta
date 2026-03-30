# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:11:56 2026

@author: Porco Rosso
"""
from functools import lru_cache
from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd

from quanta.libs._flow._main import __instance__
from quanta.config import settings
config = settings('flow').cap

__all__ = ['Series', 'DataFrame']

class Series(pd.Series):
    """
    ===========================================================================
    Custom pandas Series for quantitative portfolio management.

    Attributes
    ----------
    _internal_names : list
        Internal names for pandas compatibility.
    _internal_names_set : set
        Internal names set for fast lookup.
    _metadata : list
        Metadata fields preserved during operations.
    _config : object
        Configuration settings for capital management.
    ---------------------------------------------------------------------------
    用于量化投资组合管理的自定义 pandas Series.

    属性
    ----
    _internal_names : list
        用于 pandas 兼容性的内部名称.
    _internal_names_set : set
        用于快速查找的内部名称集合.
    _metadata : list
        在操作期间保留的元数据字段.
    _config : object
        资金管理的配置设置.
    ---------------------------------------------------------------------------
    """
    _internal_names = pd.Series._internal_names + []
    _internal_names_set = set(_internal_names)
    _metadata = pd.Series._metadata  + [f'_{i}' for i in config.recommand_settings.keys()]
    _config = config

    def __repr__(self) -> str:
        """Return a string representation of the Series | 返回 Series 的字符串表示形式"""
        x = super().__repr__()
        x = x + '\nstate: %s, unit: %s, \ncount: %s, cash: %s, \nis_adj: %s' %(self.state, self.unit, len(self), round(self.cash, 3), self.is_adj)
        return x

    @property
    def _constructor(self):
        """Internal pandas constructor | 内部 pandas 构造函数"""
        return Series

    @property
    def _constructor_sliced(self):
        """Internal pandas sliced constructor | 内部 pandas 切片构造函数"""
        return Series

    @classmethod
    def set_option(cls, dict_obj: dict) -> None:
        """Update class configuration settings | 更新类配置设置"""
        try:
            cls._config.merge_update(dict_obj)
        except Exception as e:
            print(f"config update failed: not exist keys:{dict_obj.keys()} {e}")

    @classmethod
    def init_option(cls) -> None:
        """Initialize configuration to default values | 将配置初始化为默认值"""
        init_config = settings('flow').cap
        cls._config.clear()
        cls._config.update(init_config)
        cls._metadata = pd.Series._metadata + [f'_{i}' for i in cls._config.recommand_settings.keys()]

    @classmethod
    @lru_cache(maxsize=16)
    def __get_values__(
        cls,
        portfolio_type: str,
        key: str,
        name: pd.Timestamp
    ) -> pd.Series:
        """Retrieves values from global instance registry | 从全局实例注册表中检索值"""
        x = __instance__[portfolio_type](key).loc[name]
        return x

    def __pos_neg_rebalance__(
        self,
        zero_adj: bool = True,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Rebalances positive and negative weights | 重新平衡正负权重"""
        total_weight = 1.0 if total_weight is None else total_weight
        x = self.copy()
        meta_values = x.values
        if zero_adj: # 平衡买卖 np.nansum is 5 more fast than pandas.sum()
            pos = meta_values > 0
            values = np.where(
                pos,
                np.nansum(meta_values[pos]),
                np.nansum(meta_values[~pos])
            )
            x.values[:] = meta_values / (values / total_weight)
        else: # 计算多空差异,以差异为权重1,再*total_weight
            x.values[:] = meta_values / (np.abs(np.nansum(meta_values)) / total_weight)
        return x

    def ___lazymem_clean__(self) -> None:
        """Cleans lazy memory attributes from instance | 从实例中清理延迟内存属性"""
        attrs = {i for i in self.__dict__.keys() if '__lazymem_' in i}
        [delattr(self, i) for i in attrs]

    def __add__(self, other: pd.Series) -> 'Series':
        """Element-wise addition with index union | 具有索引并集的逐元素加法"""
        if isinstance(other, (Series, pd.Series)):
            new_name =  max(
                pd.to_datetime(self.name) if self.name else pd.Timestamp.min,
                pd.to_datetime(other.name) if other.name else pd.Timestamp.min
            )
            index = self.index.union(other.index)
            x = self.reindex(index, fill_value=0).astype('float64')
            other = other.reindex(index, fill_value=0).astype('float64')
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit isnot match...')
            x = super(Series, x).__add__(other).astype('float64')
            x.name = new_name
            x.cash = x.cash + getattr(other, 'cash', 0)
        else:
            x = super().__add__(other)
        return x

    def __radd__(self, other: pd.Series) -> 'Series':
        """Reflected addition | 反向加法"""
        return self.__add__(other)

    def __sub__(self, other: pd.Series) -> 'Series':
        """Element-wise subtraction with index union | 具有索引并集的逐元素减法"""
        if isinstance(other, (Series, pd.Series)):
            new_name =  max(
                pd.to_datetime(self.name) if self.name else pd.Timestamp.min,
                pd.to_datetime(other.name) if other.name else pd.Timestamp.min
            )
            index = self.index.union(other.index)
            x = self.reindex(index, fill_value=0).astype('float64')
            other = other.reindex(index, fill_value=0).astype('float64')
            if getattr(other, 'unit', self.unit) != self.unit:
                raise ValueError('<WARNING>: portoflio unit isnot match...')
            x = super(Series, x).__sub__(other).astype('float64')
            x.name = new_name
            x.cash = x.cash - getattr(other, 'cash', 0)
        else:
            x = super().__sub__(other)
        return x

    def __rsub__(self, other: pd.Series) -> 'Series':
        """Reflected subtraction | 反向减法"""
        return self.__sub__(other)

    def __mul__(self, others: Union[int, float, np.number, pd.Series]) -> 'Series':
        """Element-wise multiplication | 逐元素乘法"""
        x = super().__mul__(others).astype('float64')
        x.name = self.name
        x.index.name = self.index.name
        return x

    def __truediv__(self, others: Union[int, float, np.number, pd.Series]) -> 'Series':
        """Element-wise division | 逐元素除法"""
        x = super().__truediv__(others).astype('float64')
        x.name = self.name
        x.index.name = self.index.name
        return x

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        dtype: Any = None,
        name: Any = None,
        copy: bool = False,
        fastpath: bool = False,
        **kwargs: Any
    ):
        """Initializes Series with data and attributes | 使用数据和属性初始化 Series"""
        params = config.recommand_settings.to_dict() | kwargs
        [setattr(self, f'_{i}',j) for i,j in params.items()]
        super().__init__(data, index, dtype, name, copy, fastpath)

    @property
    def __zero_check__(self) -> bool:
        """Checks if all values are zero | 检查所有值是否为零"""
        if not hasattr(self, '_internal_zero_check_result'):
            self._internal_zero_check_result = (not np.any(self.values))
        return self._internal_zero_check_result

    @property
    def __zero_bigger__(self) -> np.ndarray:
        """Checks which values are greater than zero | 检查哪些值大于零"""
        if not hasattr(self, '_internal_zero_bigger_result'):
            self._internal_zero_bigger_result = self.values > 0
        return self._internal_zero_bigger_result

    def __weight_to_weight__(
        self,
        zero_adj: bool = False,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Converts weight to weight with rebalance | 转换权重并重新平衡"""
        total_weight = 1 if total_weight is None else total_weight
        x = (
            self.copy()
            if self.__zero_check__ else
            self.__pos_neg_rebalance__(zero_adj, total_weight)
        )
        x._unit = 'weight'
        return x

    def __weight_to_assets__(
        self,
        cash: float,
        zero_adj: bool = False,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Converts weight to asset values | 将权重转换为资产值"""
        total_weight = 1 if total_weight is None else total_weight
        x = (
            self.copy()
            if self.__zero_check__ else
            self.__pos_neg_rebalance__(zero_adj,  total_weight*cash)
        )
        x._unit = 'assets'
        return x

    def __weight_to_share__(
        self,
        cash: float,
        zero_adj: bool = False,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Converts weight to share counts | 将权重转换为持股数量"""
        total_weight = 1 if total_weight is None else total_weight
        if self.__zero_check__:
            x = self.copy()
        else:
            x = self.__pos_neg_rebalance__(zero_adj, total_weight * cash)
            x.values[:] = x.values / self.current().values
        x._unit = 'share'
        return x

    def __assets_to_weight__(
        self,
        zero_adj: bool = False,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Converts assets to weights | 将资产转换为权重"""
        total_weight = 1 if total_weight is None else total_weight
        x = (
            self.copy()
            if self.__zero_check__ else
            self.__pos_neg_rebalance__(zero_adj, total_weight)
        )
        x._unit = 'weight'
        return x

    def __assets_to_assets__(self, **kwargs: Any) -> 'Series':
        """Identity conversion for assets | 资产到资产的恒等转换"""
        return self.copy()

    def __assets_to_share__(self, **kwargs: Any) -> 'Series':
        """Converts assets to share counts | 将资产转换为持股数量"""
        x = self.copy()
        x.values[:] = x.values / self.current().values
        x._unit = 'share'
        return x

    def __share_to_weight__(
        self,
        zero_adj: bool = False,
        total_weight: float = None,
        **kwargs: Any
    ) -> 'Series':
        """Converts shares to weights | 将持股数量转换为权重"""
        total_weight = 1 if total_weight is None else total_weight
        x = self.copy()
        if not self.__zero_check__:
            x.values[:] = x.values * self.current().values
            x = x.__assets_to_weight__(zero_adj, total_weight)
        x._unit = 'weight'
        return x

    def __share_to_assets__(self, **kwargs: Any) -> 'Series':
        """Converts shares to asset values | 将持股数量转换为资产值"""
        x = self.copy()
        x.values[:] = x.values * self.current().values
        x._unit = 'assets'
        return x

    def __share_to_share__(self, **kwargs: Any) -> 'Series':
        """Identity conversion for shares | 持股数量到持股数量的恒等转换"""
        return self.copy()

    @property
    def portfolio_type(self) -> str:
        """Returns portfolio type from index name | 从索引名称返回投资组合类型"""
        return self.index.name.split('_')[0]

    @property
    def name(self) -> pd.Timestamp:
        """Returns trade date | 返回交易日期"""
        return self._name

    @name.setter
    def name(self, trade_dt: pd.Timestamp) -> None:
        """Sets trade date and adjusts shares if necessary | 设置交易日期并在必要时调整持股"""
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
                ).values
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = post_adj.index.get_indexer(self.index)
            post_adj = post_adj.iloc[self._internal_hash_code_result].values
            self.values[:] = self.values * post_adj
        self.___lazymem_clean__()
        self._name = trade_dt

    @property
    def cash(self) -> float:
        """Calculates or returns portfolio cash | 计算或返回投资组合现金"""
        if self.state == 'settle':
            return self._cash
        elif self.state in ['signal', 'weight']:
            return 0
        else:
            if self.unit == 'share':
                x = np.nansum(self.values * self.current().values) * -1
                return x
            else:
                x = np.nansum(self.values) * -1
                return x

    @cash.setter
    def cash(self, v: float) -> None:
        """Sets portfolio cash | 设置投资组合现金"""
        self._cash = v

    @property
    def is_adj(self) -> bool:
        """Returns adjustment state | 返回调整状态"""
        return self._is_adj

    @property
    def state(self) -> str:
        """Returns current state | 返回当前状态"""
        return self._state

    @property
    def unit(self) -> str:
        """Returns current unit | 返回当前单位"""
        return self._unit

    def current(self, reindex: bool = True) -> pd.Series:
        """Retrieves current market prices | 检索当前市场价格"""
        key = self._price_mapping.get(self._state) + ('_adj' if self.is_adj else '')
        x = self.__get_values__(self.portfolio_type, key, self.name)
        if not hasattr(self, '_internal_hash_code_result'):
            self._internal_hash_code_result = x.index.get_indexer(self.index)
        if reindex:
            x = x.iloc[self._internal_hash_code_result]
        return x

    def weight(self) -> 'Series':
        """Converts portfolio to weights | 将投资组合转换为权重"""
        x = getattr(self, f"__{self.unit}_to_weight__")(copy=True)
        x._unit = 'weight'
        return x

    def assets(self, cash: float = None) -> 'Series':
        """Converts portfolio to assets | 将投资组合转换为资产值"""
        x = getattr(self, f"__{self.unit}_to_assets__")(cash=cash, copy=True)
        x._unit = 'assets'
        return x

    def share(self, cash: float = None) -> 'Series':
        """Converts portfolio to shares | 将投资组合转换为持股数量"""
        x = getattr(self, f"__{self.unit}_to_share__")(cash=cash, copy=True)
        x._unit = 'share'
        return x

    def signal(self, signal_adj: int = 0) -> 'Series':
        """Sets state to signal | 将状态设置为信号"""
        if self.state != 'signal':
            x = self.f.day_shift(signal_adj)
            x._state = 'signal'
        else:
            x = self.copy()
        return x

    def order(self, signal_adj: int = 1) -> 'Series':
        """Sets state to order | 将状态设置为订单"""
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.day_shift(signal_adj)
        x._state = 'order'
        return x

    def trade(self, signal_adj: int = 1) -> 'Series':
        """Sets state to trade | 将状态设置为交易"""
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.day_shift(signal_adj)
        x._state = 'trade'
        return x

    def settle(self, signal_adj: int = 1) -> 'Series':
        """Sets state to settle | 将状态设置为结算"""
        if self.state != 'signal':
            x = self.copy()
        else:
            x = self.f.day_shift(signal_adj)
        x._state = 'settle'
        return x

    def to(self, unit_or_state: str, **kwargs: Any) -> 'Series':
        """Generic conversion method | 通用转换方法"""
        return getattr(self, unit_or_state)(**kwargs)

    def enadj(self) -> 'Series':
        """Applies price adjustment factor | 应用价格复权因子"""
        x = self.copy()
        if (x.unit == 'share') and not x._is_adj:
            post_adj =  x.__get_values__(
                x.portfolio_type, config.post_factor, x.name
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = post_adj.index.get_indexer(self.index)
            post_adj = post_adj.iloc[self._internal_hash_code_result].values
            x.values[:] = self.values / post_adj
        x._is_adj = True
        return x

    def unadj(self) -> 'Series':
        """Removes price adjustment factor | 移除价格复权因子"""
        x = self.copy()
        if (x.unit == 'share') and x._is_adj:
            post_adj =  x.__get_values__(
                x.portfolio_type, config.post_factor, x.name
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = post_adj.index.get_indexer(self.index)
            post_adj = post_adj.iloc[self._internal_hash_code_result].values
            x.values[:] = self.values * post_adj
        x._is_adj = False
        return x

    def entrade(self) -> 'Series':
        """Filters tradable instruments | 过滤可交易标的"""
        if not hasattr(self, '_internal_entrade_result'):
            tradestatus = (
                ~self.__get_values__(self.portfolio_type, config.tradestatus, self.name)
                .astype('bool')
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = tradestatus.index.get_indexer(self.index)
            tradestatus = tradestatus.iloc[self._internal_hash_code_result].values
            self._internal_entrade_result = tradestatus
        return self[self._internal_entrade_result]

    def untrade(self) -> 'Series':
        """Filters non-tradable instruments | 过滤不可交易标的"""
        if not hasattr(self, '_internal_entrade_result'):
            tradestatus = (
                ~self.__get_values__(self.portfolio_type, config.tradestatus, self.name)
                .astype('bool')
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = tradestatus.index.get_indexer(self.index)
            tradestatus = tradestatus.iloc[self._internal_hash_code_result].values
            self._internal_entrade_result = tradestatus
        return self[~self._internal_entrade_result]

    def enbuy(self) -> 'Series':
        """Filters instruments available for buying | 过滤可买入标的"""
        if not hasattr(self, '_internal_enbuy_result'):
            buy = (
                1 -
                self.__get_values__(self.portfolio_type, config.trade_price, self.name) /
                self.__get_values__(self.portfolio_type, config.high_limit, self.name).values
                >=
                self._untrade_limit
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = buy.index.get_indexer(self.index)
            buy = buy.iloc[self._internal_hash_code_result].values
            self._internal_enbuy_result = buy
        x = self[self._internal_enbuy_result & self.__zero_bigger__]
        return x

    def unbuy(self) -> 'Series':
        """Filters instruments restricted from buying | 过滤禁止买入标的"""
        if not hasattr(self, '_internal_enbuy_result'):
            buy = (
                1 -
                self.__get_values__(self.portfolio_type, config.trade_price, self.name) /
                self.__get_values__(self.portfolio_type, config.high_limit, self.name).values
                >=
                self._untrade_limit
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = buy.index.get_indexer(self.index)
            buy = buy.iloc[self._internal_hash_code_result].values
            self._internal_enbuy_result = buy
        x = self[~self._internal_enbuy_result & self.__zero_bigger__]
        return x

    def ensell(self) -> 'Series':
        """Filters instruments available for selling | 过滤可卖出标的"""
        if not hasattr(self, '_internal_ensell_result'):
            sell = (
                1 -
                self.__get_values__(self.portfolio_type, config.low_limit, self.name) /
                self.__get_values__(self.portfolio_type, config.trade_price, self.name).values
                >=
                self._untrade_limit
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = sell.index.get_indexer(self.index)
            sell = sell.iloc[self._internal_hash_code_result].values
            self._internal_ensell_result = sell
        x = self[self._internal_ensell_result & ~self.__zero_bigger__]
        return x

    def unsell(self) -> 'Series':
        """Filters instruments restricted from selling | 过滤禁止卖出标的"""
        if not hasattr(self, '_internal_ensell_result'):
            sell = (
                1 -
                self.__get_values__(self.portfolio_type, config.low_limit, self.name) /
                self.__get_values__(self.portfolio_type, config.trade_price, self.name).values
                >=
                self._untrade_limit
            )
            if not hasattr(self, '_internal_hash_code_result'):
                self._internal_hash_code_result = sell.index.get_indexer(self.index)
            sell = sell.iloc[self._internal_hash_code_result].values
            self._internal_ensell_result = sell
        x = self[~self._internal_ensell_result & ~self.__zero_bigger__]
        return x

    def trade_cost(self, pct: float = None) -> float:
        """Calculates total trading costs | 计算总交易成本"""
        x = np.nansum(np.abs(self.assets().values))
        x = x * (self._trade_cost if pct is None else pct)
        return x

    def total_assets(self) -> float:
        """Calculates total portfolio asset value | 计算投资组合总资产价值"""
        x = np.nansum(self.assets().values)
        if np.isnan(x):
            x = 0
        return x + self.cash

class DataFrame(pd.DataFrame):
    """
    ===========================================================================
    Custom pandas DataFrame for quantitative portfolio management.

    Attributes
    ----------
    _internal_names : list
        Internal names for pandas compatibility.
    _internal_names_set : set
        Internal names set for fast lookup.
    _metadata : list
        Metadata fields preserved during operations.
    ---------------------------------------------------------------------------
    用于量化投资组合管理的自定义 pandas DataFrame.

    属性
    ----
    _internal_names : list
        用于 pandas 兼容性的内部名称.
    _internal_names_set : set
        用于快速查找的内部名称集合.
    _metadata : list
        在操作期间保留的元数据字段.
    ---------------------------------------------------------------------------
    """
    _internal_names = pd.DataFrame._internal_names + []
    _internal_names_set = set(_internal_names)
    _metadata = pd.DataFrame._metadata

    @property
    def _constructor(self):
        """Internal pandas constructor | 内部 pandas 构造函数"""
        return DataFrame

    @property
    def _constructor_sliced(self):
        """Internal pandas sliced constructor | 内部 pandas 切片构造函数"""
        return Series
