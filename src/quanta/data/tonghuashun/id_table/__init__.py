from typing import Literal, Optional, Sequence

from quanta.config import settings as _settings

from .main import main as _class_obj

_config = _settings('data').tables.ths_table

__all__ = ['daily']


def daily(
    codes: Optional[Sequence[str]] = None,
    if_exists: Literal['append', 'replace'] = 'append',
    limit: Optional[int] = 50
) -> None:
    """
    ===========================================================================
    Runs configured Tonghuashun ID-table update pipelines.

    Parameters
    ----------
    codes : Optional[Sequence[str]]
        Stock codes to update. None uses the complete stock universe.
    if_exists : Literal['append', 'replace']
        Append new report IDs or rebuild the table.
    limit : Optional[int]
        Maximum latest reports inspected per stock. None means all.
    ---------------------------------------------------------------------------
    运行配置的同花顺 ID 表更新流水线.

    参数
    ----
    codes : Optional[Sequence[str]]
        需要更新的股票代码. None 使用完整股票池.
    if_exists : Literal['append', 'replace']
        追加新研报 ID 或重建数据表.
    limit : Optional[int]
        每只股票检查的最新研报上限. None 表示全部.
    ---------------------------------------------------------------------------
    """
    for table_config in _config.values():
        instance_obj = _class_obj(**table_config)
        instance_obj.daily(
            codes=codes,
            if_exists=if_exists,
            limit=limit
        )
