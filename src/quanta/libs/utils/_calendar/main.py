# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:29:07 2026

@author: Porco Rosso
"""
from functools import lru_cache
from typing import Optional, Union

import pandas as pd
from ....config import settings, login_info
from ...db.main import main as db

class main():
    """
    ===========================================================================
    Trading calendar utilities providing calendar days and trade days with
    configurable time biases, backed by JoinQuant or a local backup table.
    ---------------------------------------------------------------------------
    交易日历工具, 提供可配置时间偏置的日历日和交易日, 由聚宽或本地备份表支持.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self,
        start: Union[str, pd.Timestamp],
        bias: str = '19 hours',
        daily_bias: Optional[str] = '15 hours',
        name: str = 'trade_dt',
        baktable: str = 'astockeodprices'
    ) -> None:
        """Initializes the calendar with start date and time biases | 用起始日期和时间偏置初始化日历"""
        self.start = pd.to_datetime(start)
        self.bias = bias
        self.daily_bias = daily_bias
        self.name = name
        self.baktable = baktable

    @lru_cache(maxsize=1)
    def _internal_calendar_days(
        self,
        start: Union[str, pd.Timestamp],
        bias: str,
        daily_bias: Optional[str],
        name: str
    ) -> pd.Series:
        """Builds the daily calendar series with the configured biases | 构建带配置偏置的每日日历序列"""
        calendar_days = pd.date_range(
            start = start,
            freq = 'd',
            end = pd.Timestamp.today() - pd.Timedelta(bias),
            name = name)
        if daily_bias is not None:
            calendar_days = calendar_days + pd.Timedelta(daily_bias)
        calendar_days = pd.Series(calendar_days, index = calendar_days)
        return calendar_days

    @property
    def calendar_days(self) -> pd.Series:
        """Returns the daily calendar series | 返回每日日历序列"""
        x = self._internal_calendar_days(self.start, self.bias, self.daily_bias, self.name)
        return x

    @lru_cache(maxsize=1)
    def _internal_trade_days(
        self,
        start: Union[str, pd.Timestamp],
        bias: str,
        daily_bias: Optional[str],
        name: str
    ) -> pd.Series:
        """Builds the trade day series from JoinQuant or the backup table | 从聚宽或备份表构建交易日序列"""
        try:
            import jqdatasdk as jq
            jq.auth(**login_info('account').joinquant)
            trade_days = pd.to_datetime(
                jq.get_trade_days(
                    start,
                    pd.Timestamp.today() - pd.Timedelta(bias)
                )
            )
            trade_days.name = name

        except Exception:
            trade_days = db().__read__(
                table = self.baktable,
                columns = name).iloc[:, 0] - pd.Timedelta(settings('data').public_keys.recommand_settings.time_bias)
            trade_days = pd.to_datetime(
                sorted(
                    trade_days[trade_days >= pd.to_datetime(start)].unique()
                )
            )
        if daily_bias is not None:
            trade_days = trade_days + pd.Timedelta(daily_bias)
        trade_days =  pd.Series(trade_days, index = trade_days)
        return trade_days

    @property
    def trade_days(self) -> pd.Series:
        """Returns the trade day series | 返回交易日序列"""
        x = self._internal_trade_days(self.start, self.bias, self.daily_bias, self.name)
        return x

    def units(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        window: Optional[int] = None,
        day_set: str = 'trade_days'
    ) -> pd.Series:
        """
        =======================================================================
        Selects a slice of days bounded by start, end, or a window of length.

        Parameters
        ----------
        start : Optional[Union[str, pd.Timestamp]]
            The start bound. Default is None.
        end : Optional[Union[str, pd.Timestamp]]
            The end bound. Default is None.
        window : Optional[int]
            The number of days to keep. Default is None.
        day_set : str
            Which day series to use ('trade_days' or 'calendar_days').
            Default is 'trade_days'.

        Returns
        -------
        pd.Series
            The selected day slice.
        -----------------------------------------------------------------------
        按 start, end 或 window 长度选取一段日期切片.

        参数
        ----
        start : Optional[Union[str, pd.Timestamp]]
            起始边界. 默认为 None.
        end : Optional[Union[str, pd.Timestamp]]
            结束边界. 默认为 None.
        window : Optional[int]
            保留的天数. 默认为 None.
        day_set : str
            使用的日期序列 ('trade_days' 或 'calendar_days'). 默认为 'trade_days'.

        返回
        ----
        pd.Series
            选取的日期切片.
        -----------------------------------------------------------------------
        """
        parameter_bool = [i is None for i in [start, end, window]]
        if all(parameter_bool) or not any(parameter_bool):
            raise ValueError("<start>, <end>, <window> must have 2 not None values and 1 None value")
        days = getattr(self, day_set)
        if start is not None:
            days = days.loc[pd.to_datetime(start):]
            if window is not None:
                days = days.iloc[:window]
        if end is not None:
            days = days.loc[:pd.to_datetime(end)]
            if window is not None:
                days = days.iloc[-window:]
        return days
