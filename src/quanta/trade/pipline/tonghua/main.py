# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:17:07 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd

from quanta.config import settings
config = settings('trade').pipline.tonghua

class main():
    """
    ===========================================================================
    Tonghua (同花顺) pipeline adapter for reading and writing trade files
    in broker-specific formats.
    ---------------------------------------------------------------------------
    同花顺管道适配器, 用于以券商特定格式读写交易文件.
    ---------------------------------------------------------------------------
    """

    def __init__(self, broker: str) -> None:
        """Initializes the pipeline with broker-specific settings | 用券商特定设置初始化管道"""
        [setattr(self, i,j) for i,j in getattr(config, broker).items()]
        self.broker = broker

    def read(self, path: str) -> pd.DataFrame:
        """
        =======================================================================
        Reads a trade file and returns the settle columns.

        Parameters
        ----------
        path : str
            The file path (with or without extension).

        Returns
        -------
        pd.DataFrame
            The settle columns of the parsed file.
        -----------------------------------------------------------------------
        读取交易文件并返回结算列.

        参数
        ----
        path : str
            文件路径 (可带或不带扩展名).

        返回
        ----
        pd.DataFrame
            解析文件的结算列.
        -----------------------------------------------------------------------
        """
        data_mapping = {
            'xls': [pd.read_csv, {'encoding': 'gbk', 'sep': '\t'}],
            'xlsx': [pd.read_excel, {}],
            'csv': [pd.read_csv, {}]
            }

        func = data_mapping.get(self.settle.dtype, None)
        if func is None:
            raise ValueError('Undefined date type...')
        else:
            if path.split('.')[-1] != self.settle.dtype:
                path = '.'.join([path, self.settle.dtype])
            x = func[0](path, **func[1])[list(self.settle.columns.values())]
            return x

    def write(self, df: pd.DataFrame, path: str) -> None:
        """
        =======================================================================
        Writes an order DataFrame to a file in the broker format.

        Parameters
        ----------
        df : pd.DataFrame
            The order DataFrame to write.
        path : str
            The output file path (with or without extension).
        -----------------------------------------------------------------------
        将订单 DataFrame 以券商格式写入文件.

        参数
        ----
        df : pd.DataFrame
            要写入的订单 DataFrame.
        path : str
            输出文件路径 (可带或不带扩展名).
        -----------------------------------------------------------------------
        """
        data_mapping = {
            'xls': ['to_excel', {}],
            'xlsx': ['to_excel', {}],
            'csv': ['to_csv', {'encoding': 'gbk'}]
            }
        func = data_mapping.get(self.order.dtype, None)
        if func is None:
            raise ValueError('Undefined date type...')
        else:
            if path.split('.')[-1] != self.order.dtype:
                path = '.'.join([path, self.order.dtype])
                df.columns = list(self.order.columns.values())
                getattr(df, func[0])(path, **func[1])






