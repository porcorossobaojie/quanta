from typing import Literal, Optional, Sequence

from .joinquant import daily as _jq_daily, minute as _jq_minute
from .tonghuashun import research_report as _ths_research_report


def daily() -> None:
    """Builds the local daily database | 构建本地日频数据库"""
    _jq_daily()


def minute() -> None:
    """Builds the local minute database | 构建本地分钟频数据库"""
    _jq_minute()


def research_report(
    codes: Optional[Sequence[str]] = None,
    if_exists: Literal['append', 'replace'] = 'append',
    limit: Optional[int] = 50,
    workers: Optional[int] = 16
) -> None:
    """
    ===========================================================================
    Updates Tonghuashun research reports for selected stocks.

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
    ---------------------------------------------------------------------------
    更新指定股票的同花顺研报.

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
    ---------------------------------------------------------------------------
    """
    _ths_research_report(
        codes=codes,
        if_exists=if_exists,
        limit=limit,
        workers=workers
    )
