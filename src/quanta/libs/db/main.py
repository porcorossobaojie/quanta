# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:59:31 2026

@author: Porco Rosso
"""

from ..db._engines.DuckDB import main as DuckDB
from ..db._engines.MySQL import main as MySQL
from ...config import settings

config = settings('libs').db
ENGINES = {'DuckDB': DuckDB, 'MySQL': MySQL}
meta = ENGINES.get(config.recommend_settings)

__all__ = ['main']


class main(meta):
    """
    ===========================================================================
    Main class for database operations, acting as a facade for different
    database types.

    This class dynamically dispatches calls to the appropriate database
    implementation (MySQL or DuckDB) based on the configured settings.
    ---------------------------------------------------------------------------
    数据库操作的主类, 作为不同数据库类型的门面.

    此类根据配置的设置动态地将调用分派给相应的数据库实现 (MySQL 或 DuckDB).
    ---------------------------------------------------------------------------
    """
    engine_type = config.recommend_settings

    def __init__(self, **kwargs) -> None:
        """Initializes the database facade with keyword attributes | 用关键字属性初始化数据库门面"""
        [setattr(self, i, j) for i, j in kwargs.items()]
