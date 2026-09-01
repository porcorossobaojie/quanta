# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:30:52 2026

@author: Porco Rosso
"""

import pandas as pd
from typing import Literal, Optional, Any
from ...db.main import main as db

MODULE_DIR = __name__.split('.')[-2]
# Registering the db main instance directly to pandas for generic access
setattr(pd, MODULE_DIR, db())
db.__env_init__()

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main:
    """
    ===========================================================================
    Pandas DataFrame accessor for database operations, allowing direct writing
    of DataFrames to the configured database.
    ---------------------------------------------------------------------------
    用于数据库操作的 Pandas DataFrame 访问器, 允许将 DataFrame 直接写入配置的数据库.
    ---------------------------------------------------------------------------
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Initializes the DataFrame accessor | 初始化 DataFrame 访问器"""
        self._obj: pd.DataFrame = pandas_obj

    def write(
        self,
        table: str,
        if_exists: Literal['fail', 'replace', 'append'] = 'append',
        index: bool = False,
        log: bool = True,
        **kwargs: Any
    ) -> None:
        """
        =======================================================================
        Writes the DataFrame to a database table.

        Parameters
        ----------
        table : str
            The name of the target table in the database.
        if_exists : Literal['fail', 'replace', 'append']
            How to behave if the table already exists. Default is 'append'.
        index : bool
            Whether to write the DataFrame index as a column. Default is False.
        log : bool
            Whether to log the operation details. Default is True.
        **kwargs : Any
            Additional keyword arguments for the database engine's write
            method.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        将 DataFrame 写入数据库表.

        参数
        ----
        table : str
            数据库中的目标表名.
        if_exists : Literal['fail', 'replace', 'append']
            如果表已存在时的处理方式. 默认为 'append'.
        index : bool
            是否将 DataFrame 索引作为一列写入. 默认为 False.
        log : bool
            是否记录操作详情. 默认为 True.
        **kwargs : Any
            传递给数据库引擎写入方法的附加关键字参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        db().__write__(self._obj, table=table, if_exists=if_exists, index=index, log=log, **kwargs)
        
    def write_parquet(
        self,
        path: str,
        file_name: Optional[str] = None,
        log: bool = True
    ) -> None:
        """
        =======================================================================
        Writes the DataFrame to a parquet file under the given path.

        Parameters
        ----------
        path : str
            The destination directory for the parquet file.
        file_name : str, optional
            The parquet file name; defaults to 'data_0' when None.
        log : bool
            Whether to log the operation details. Default is True.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        将 DataFrame 写入指定路径下的 parquet 文件.

        参数
        ----
        path : str
            parquet 文件的输出目录.
        file_name : str, optional
            parquet 文件名; 为 None 时默认为 'data_0'.
        log : bool
            是否记录操作详情. 默认 True.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        db().__write_parquet__(self._obj, path=path, file_name=file_name, log=log)
