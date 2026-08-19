# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:13:20 2026

@author: Porco Rosso
"""
import shutil
import os
import jqdatasdk as jq
from typing import Any, Literal
import pandas as pd
from pathlib import Path
from quanta.data.joinquant.meta.mins import main as meta
from quanta.config import settings

config = settings('data')

class main(meta):
    """
    ===========================================================================
    Main class for handling trade date table data from JoinQuant.

    This class extends the meta class to provide specific data processing
    and daily update functionalities for trade date related tables.
    ---------------------------------------------------------------------------
    处理 JoinQuant 交易日期表数据的主类.

    此类扩展了元类, 为交易日期相关表提供特定的数据处理和每日更新功能.
    ---------------------------------------------------------------------------
    """
    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        """
        =======================================================================
        Overrides the base pipeline to include automatic return calculation.
        It uses (adjusted) close prices to derive returns and filters out rows
        containing only metadata.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for the data extraction process.

        Returns
        -------
        pd.DataFrame
            The fully processed and augmented DataFrame.
        -----------------------------------------------------------------------
        重写基类流水线以包含自动收益率计算. 它使用 (复权) 收盘价来推导收益率, 并
        过滤掉仅包含元数据的行.

        参数
        ----
        **kwargs : Any
            数据提取过程的关键字参数.

        返回
        ----
        pd.DataFrame
            完全处理和增强后的 DataFrame.
        -----------------------------------------------------------------------
        """
        df = super().pipeline(**kwargs)

        # construct returns of portfolio by calculated with close price(first use adj price if have)
        if isinstance(self.columns_information, dict):
            ret_key = self.columns_information.get('returns', None)
        else:
            x = eval(f"jq.get_table_info({self.columns_information})")
            x.iloc[:, 0] = x.iloc[:, 0].replace(config.tables.transform | {'code': self.code})
            x.iloc[:, 2] = x.iloc[:, 2].replace({'date': 'datetime', 'DATE': 'datetime'})
            x = x.set_index(x.columns[0]).iloc[:, [1, 0]].T.to_dict('list')
            ret_key = x.get('returns', None)
        if ret_key is not None:
            ret_key = list(ret_key.keys())[0]
            try:
                df[ret_key] = df['close_adj'] / df['preclose_adj'] - 1
            except Exception:
                df[ret_key] = df['close'] / df['preclose'] - 1
        df = df[df.drop([self.trade_dt, self.code], axis=1, errors='ignore').notnull().any(axis=1)]
        df.columns.name = 'values'
        df.columns = pd.CategoricalIndex(df.columns)
        df[self.code] = pd.CategoricalIndex(df[self.code])
        df = df.set_index([self.trade_dt, self.code]).unstack(self.code).T.sort_index()
        return df

    def daily(self, if_exists: Literal['append', 'replace'] = 'append') -> None:
        """
        =======================================================================
        Performs daily updates for trade-date based tables. Special handling
        is applied to 'astocklisting' which is fully refreshed daily. For
        others, it performs incremental updates based on the last trade date.

        Parameters
        ----------
        if_exists : Literal['append', 'replace']
            Strategy when the table already exists. Default is 'append'.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        执行基于交易日期的表的每日更新. 对每日全量刷新的 'astocklisting' 进行
        特殊处理. 对于其他表, 根据最后一个交易日执行增量更新.

        参数
        ----
        if_exists : Literal['append', 'replace']
            当表已存在时的策略. 默认为 'append'.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        table_path = Path(self.path) / self.parquet / self.table
        if if_exists == 'replace':
            if os.path.exists(table_path):
                shutil.rmtree(table_path)
        table_path.mkdir(parents=True, exist_ok=True)

        id_key = [p for p in table_path.iterdir() if p.is_dir()]
        id_key = self.date_start if not len(id_key) else max(id_key).__str__().split('date=')[-1]
        days = self.trade_days[self.trade_days > id_key]

        for i in days:
            if jq.get_query_count()['spare'] > 10000000:
                print(f"Update to date: {i.date()}; Query count left: {jq.get_query_count()['spare']}")
                df = self.pipeline(start_date= str(i.date()), end_date=str(i.date()))
                df.db.write_parquet(path=table_path/f"date={i.date()}", log=True)
            else:
                print("not enough token...")
                break

