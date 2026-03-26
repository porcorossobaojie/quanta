# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:42:22 2026

@author: Porco Rosso
"""

import pandas as pd
from typing import Any, List, Optional
from pathlib import Path

from quanta import flow
from quanta.trade import pipline
from quanta.config import settings

char = settings('trade').strategy_001.BJ_13611823855
config = settings('data').public_keys.recommand_settings

__all__ = ['main']

class main:
    """
    ===========================================================================
    Main class for trading account management, handling order paths,
    settlement processing, and pipeline integration.
    ---------------------------------------------------------------------------
    交易账户管理的主类, 处理订单路径, 结算处理和流水线集成.
    ---------------------------------------------------------------------------
    """
    system = settings('trade').system

    def __init__(self, **kwargs: Any):
        """Initializes account settings | 初始化账户设置"""
        [setattr(self, f"_{i}", j) for i,j in kwargs.items()]
        self.__env_init__()

    @property
    def pipline(self) -> Any:
        """Returns the trading pipeline instance | 返回交易流水线实例"""
        x = getattr(pipline, self._pipline)
        x = x(broker=self._broker)
        return x

    @property
    def portfolio_type(self) -> str:
        """Returns the portfolio type (e.g., 'astock') | 返回组合类型"""
        x = getattr(self, '_portfolio_type', config.key.astock_code.split('_')[0])
        return x

    @property
    def __order_path__(self) -> Path:
        """Constructs the path for order files | 构建订单文件路径"""
        x = Path(self.system.path) / self._strategy / self._name / self.system.order
        return x

    @property
    def __settle_path__(self) -> Path:
        """Constructs the path for settlement files | 构建结算文件路径"""
        x = Path(self.system.path) / self._strategy / self._name / self.system.settle
        return x

    def __env_init__(self) -> None:
        """Creates necessary directory structures | 创建必要的目录结构"""
        self.__order_path__.mkdir(parents=True, exist_ok=True)
        self.__settle_path__.mkdir(parents=True, exist_ok=True)

    def __get_files_name__(self, path: Path) -> List[str]:
        """Retrieves file names in a directory | 获取目录中的文件名"""
        files = [f.name for f in path.iterdir() if f.is_file()]
        return files

    def settle(self, file_name_by_date: Optional[str] = None) -> pd.Series:
        """
        =======================================================================
        Processes settlement data for a specific date or the most recent date.

        Parameters
        ----------
        file_name_by_date : Optional[str]
            The settlement file name (usually a date string). If None, uses
            the most recent file.

        Returns
        -------
        pd.Series
            A Series containing settled positions, indexed by standardized
            asset codes.
        -----------------------------------------------------------------------
        处理特定日期或最近日期的结算数据.

        参数
        ----
        file_name_by_date : Optional[str]
            结算文件名 (通常为日期字符串). 如果为 None, 则使用最近的文件.

        返回
        ----
        pd.Series
            包含已结算持仓的 Series, 以标准化的资产代码作为索引.
        -----------------------------------------------------------------------
        """
        if file_name_by_date is None:
            file_name_by_date = max(self.__get_files_name__(self.__settle_path__)).split('.')[0]
        x = self.pipline.read(str(self.__settle_path__ / file_name_by_date))
        x = x.set_index(x.columns[0])
        x = x.iloc[:, 0]
        x.index = getattr(flow, self.portfolio_type).code_standard(x.index)
        x.name = pd.to_datetime(file_name_by_date) + pd.Timedelta(config.key.time_bias)
        x = pd.f.Series(x, unit='share', state = 'settle', is_adj=False)
        x = x[x.index.notnull()].astype('float64')
        return x
