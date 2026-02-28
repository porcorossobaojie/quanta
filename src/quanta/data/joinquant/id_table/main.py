# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:11:28 2026

@author: Porco Rosso
"""

from typing import Any, Literal

import pandas as pd

from quanta.libs.data.joinquant.meta.main import main as meta

from quanta.config import settings
config = settings('data')

class main(meta):
    """
    ===========================================================================

    Main class for handling announcement date table data from JoinQuant.

    This class extends the meta class to provide specific data processing
    and daily update functionalities for announcement date related tables.

    ---------------------------------------------------------------------------

    处理 JoinQuant 公告日期表数据的主类。

    此类扩展了元类，为公告日期相关表提供特定的数据处理和每日更新功能。

    ---------------------------------------------------------------------------
    """

    def create_table_afundshare(self, **kwargs):
        columns = {self.trade_dt if i == self.report_period else i:j for i,j in self.columns.items()}
        self.create_table(columns = columns)
    
    def __data_standard_afundshare__(self, df, **kwargs):
        df = self.__data_standard__(df, date=self.report_period)
        df = df.rename({self.report_period: self.trade_dt}, axis=1)
        df[self.trade_dt] = pd.to_datetime(df[self.trade_dt]) + pd.Timedelta(config.tables.recommand_settings.time_bias)
        return df

    def daily(self, if_exists: Literal['append', 'replace'] = 'append') -> None:
        """
        ===========================================================================

        Performs daily updates for the announcement date table.

        This method handles the logic for appending or replacing data based on
        the `if_exists` parameter, ensuring the table is up-to-date.

        Parameters
        ----------
        if_exists : Literal['append', 'replace'], optional
            Determines how to handle existing data. 'append' adds new data,
            'replace' drops the table and recreates it before adding data.
            Defaults to 'append'.

        ---------------------------------------------------------------------------

        执行公告日期表的每日更新。

        此方法根据 `if_exists` 参数处理追加或替换数据的逻辑，确保表格是最新的。

        参数
        ----------
        if_exists : Literal['append', 'replace'], optional
            确定如何处理现有数据。'append' 添加新数据，'replace' 在添加数据前
            删除并重新创建表格。默认为 'append'。

        ---------------------------------------------------------------------------
        """
        if if_exists == 'replace':
            self.drop_table()

        if not self.table_exist():
            getattr(self,  f"create_table_{self.table}", self.create_table)()

        id_key = self.__find_max_of_exist_table__(self.id_key)
        df = self.pipeline(id_key=id_key)
        self.__write__(df, log=True)
        while len(df):
            id_key = df[self.id_key].max()
            df = self.pipeline(id_key=id_key)
            self.__write__(df, log=True)



'''

self = main(**config.tables.id_table.astockstatus)
self.daily()

 '''
