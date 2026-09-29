# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:49:47 2026

@author: Porco Rosso
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional, Sequence

import pandas as pd

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
        limit: Optional[int] = 50,
        workers: Optional[int] = None
    ) -> None:
        """
        =======================================================================
        Performs an incremental update using report-ID set differences.
        Each stock is fetched, standardized, and persisted on its own, so
        progress is durable even if a long universe-wide run is interrupted.

        Parameters
        ----------
        codes : Optional[Sequence[str]]
            Stock codes to update. None uses the complete stock universe.
        if_exists : Literal['append', 'replace']
            Append new report IDs or rebuild the table.
        limit : Optional[int]
            Maximum latest reports inspected per stock. None means all.
        workers : Optional[int]
            Concurrent stock fetchers. None uses the configured default.
        -----------------------------------------------------------------------
        使用研报 ID 集合差分执行增量更新.
        每只股票抓取, 标准化后立即单独入库, 因此即使长时间全量任务被中断,
        已完成股票的进度也会持久化.

        参数
        ----
        codes : Optional[Sequence[str]]
            需要更新的股票代码. None 使用完整股票池.
        if_exists : Literal['append', 'replace']
            追加新研报 ID 或重建数据表.
        limit : Optional[int]
            每只股票检查的最新研报上限. None 表示全部.
        workers : Optional[int]
            并发抓取的股票线程数. None 使用配置默认值.
        -----------------------------------------------------------------------
        """
        if if_exists == 'replace':
            self.drop_table()
        if not self.table_exist():
            self.create_table()

        codes = list(self._stock if codes is None else codes)
        id_keys = self.__existing_ids__(codes)
        total = len(codes)
        workers = getattr(self, 'workers', 4) if workers is None else workers
        workers = max(1, int(workers))
        written = 0
        finished = 0

        def persist(code: str, df: pd.DataFrame) -> None:
            """Persists one completed stock immediately | 单只股票完成后立即入库"""
            nonlocal written, finished
            finished += 1
            if not df.empty:
                self.__write__(df, log=False)
                written += len(df)
                print(
                    f'[quanta] {self.table}: {finished}/{total} <{code}> '
                    f'+{len(df)} records (total {written})'
                )
            elif finished % 200 == 0:
                print(
                    f'[quanta] {self.table}: progress {finished}/{total}, '
                    f'{written} new records'
                )

        if workers == 1:
            for code in codes:
                try:
                    df = self.pipeline(codes=[code], id_keys=id_keys, limit=limit)
                except Exception as error:
                    print(f'[quanta] {self.table} failed <{code}>: {error}')
                    finished += 1
                    continue
                persist(code, df)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.pipeline,
                        codes=[code],
                        id_keys=id_keys,
                        limit=limit
                    ): code
                    for code in codes
                }
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        df = future.result()
                    except Exception as error:
                        print(f'[quanta] {self.table} failed <{code}>: {error}')
                        finished += 1
                        continue
                    persist(code, df)

        print(
            f'[quanta] {self.table}: finished, {written} new records '
            f'from {total} stocks.'
        )
