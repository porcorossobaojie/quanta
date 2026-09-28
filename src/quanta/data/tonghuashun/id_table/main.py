# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:49:47 2026

@author: Porco Rosso
"""

from typing import Literal, Optional, Sequence

from ..meta.main import main as meta


class main(meta):
    """
    ===========================================================================
    Main class for ID-based Tonghuashun research report updates.
    ---------------------------------------------------------------------------
    基于 ID 更新同花顺研报的主类.
    ---------------------------------------------------------------------------
    """

    def daily(
        self,
        codes: Optional[Sequence[str]] = None,
        if_exists: Literal['append', 'replace'] = 'append',
        limit: Optional[int] = 50
    ) -> None:
        """
        =======================================================================
        Performs an incremental update using report-ID set differences.

        Parameters
        ----------
        codes : Optional[Sequence[str]]
            Stock codes to update. None uses the complete stock universe.
        if_exists : Literal['append', 'replace']
            Append new report IDs or rebuild the table.
        limit : Optional[int]
            Maximum latest reports inspected per stock. None means all.
        -----------------------------------------------------------------------
        使用研报 ID 集合差分执行增量更新.

        参数
        ----
        codes : Optional[Sequence[str]]
            需要更新的股票代码. None 使用完整股票池.
        if_exists : Literal['append', 'replace']
            追加新研报 ID 或重建数据表.
        limit : Optional[int]
            每只股票检查的最新研报上限. None 表示全部.
        -----------------------------------------------------------------------
        """
        if if_exists == 'replace':
            self.drop_table()
        if not self.table_exist():
            self.create_table()

        codes = self._stock if codes is None else codes
        id_keys = self.__existing_ids__(codes)
        df = self.pipeline(codes=codes, id_keys=id_keys, limit=limit)
        if not df.empty:
            self.__write__(df, log=True)
