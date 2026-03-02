# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:30:52 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Literal, Any
from quanta.libs.db.main import main as db

MODULE_DIR = __name__.split('.')[-2]
# Registering the db main instance directly to pandas for generic access
setattr(pd, MODULE_DIR, db())


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

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj: pd.DataFrame = pandas_obj

    def write(
        self,
        table_name: str,
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
        table_name : str
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
        table_name : str
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
        db().write(self._obj, table_name=table_name, if_exists=if_exists, index=index, log=log, **kwargs)
