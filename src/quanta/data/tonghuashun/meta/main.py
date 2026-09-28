# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:49:47 2026

@author: Porco Rosso
"""

from typing import Any, Sequence, Set

import jqdatasdk as jq
import numpy as np
import pandas as pd
import requests

from ....config import settings
from ....libs.db.main import main as db
from ._common import common

config = settings('data')


class main(
    common,
    db,
    type('recommend_settings', (), config.tables.recommend_settings.key)
):
    """
    ===========================================================================
    Base metadata and connection class for Tonghuashun report extraction.
    ---------------------------------------------------------------------------
    用于同花顺研报提取的基础元数据及连接类.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        =======================================================================
        Initializes the database environment and reusable HTTP session.

        Parameters
        ----------
        **kwargs : Any
            Initial configuration and table parameters.
        -----------------------------------------------------------------------
        初始化数据库环境及可复用 HTTP 会话.

        参数
        ----
        **kwargs : Any
            初始配置及表参数.
        -----------------------------------------------------------------------
        """
        super().__init__(**kwargs)
        self.__env_init__()
        self.session = requests.Session()
        self.session.headers.update(dict(self.headers))
        self._last_request_at = None
        self._stock = jq.get_all_securities('stock', date=None).index.tolist()

    def __data_standard__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Standardizes source columns, security codes, and announcement times.

        Parameters
        ----------
        df : pd.DataFrame
            Raw source-shaped report records.
        **kwargs : Any
            Additional pipeline arguments.

        Returns
        -------
        pd.DataFrame
            Standardized report records.
        -----------------------------------------------------------------------
        标准化源字段, 证券代码及公告时间.

        参数
        ----
        df : pd.DataFrame
            源格式研报记录.
        **kwargs : Any
            流水线附加参数.

        返回
        ----
        pd.DataFrame
            标准化研报记录.
        -----------------------------------------------------------------------
        """
        df = self.__columns_rename__(df)
        if df.empty:
            return df
        df[self.code] = df[self.code].map(self.__standard_code__)
        df[self.ann_dt] = pd.to_datetime(df[self.ann_dt]) + pd.Timedelta(
            config.tables.recommend_settings.time_bias
        )
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        return df

    def __existing_ids__(self, codes: Sequence[str]) -> Set[int]:
        """Reads persisted report IDs for selected stocks | 读取指定股票已持久化的研报 ID"""
        if not self.table_exist():
            return set()
        standard_codes = [self.__standard_code__(code) for code in codes]
        quoted_codes = ', '.join(f"'{code}'" for code in standard_codes)
        df = self.__read__(
            columns=self.id_key,
            where=f'{self.code} IN ({quoted_codes})',
            show_time=False
        )
        return set(pd.to_numeric(df[self.id_key]).astype('int64').tolist())

    def table_exist(self) -> bool:
        """Checks if the current report table exists | 检查当前研报表是否存在"""
        return super().__table_exist__()

    def drop_table(self, **kwargs: Any) -> None:
        """Drops the current report table | 删除当前研报表"""
        parameters = self.__parameters__({'log': True}, kwargs)
        super().__drop_table__(**parameters)

    def create_table(self, **kwargs: Any) -> None:
        """Creates the current report table | 创建当前研报表"""
        parameters = self.__parameters__(
            {
                'columns': self.columns,
                'keys': [self.code, self.ann_dt, self.id_key],
                'log': True
            },
            kwargs
        )
        super().__create_table__(**parameters)
