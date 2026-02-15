# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:43:22 2026

@author: Porco Rosso
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import jqdatasdk as jq

from quanta.libs.utils import merge_dicts
from quanta.config import settings, login_info
from quanta.libs.db.main import main as db

#jq.auth(**login_info('account').joinquant)
config = settings('data')


class main(db, type('recommand_settings', (), config.tables.recommand_settings.key)):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__env_init__()
        self._stock = jq.get_all_securities('stock', date=None).index.tolist()
        _fund = jq.get_all_securities('fund', date=None)
        self._fund = _fund[_fund.iloc[:, -1] == 'etf'].index.tolist()
        self._index = jq.get_all_securities('index', date=None).index.tolist()
        _trade_days = pd.to_datetime(jq.get_trade_days('2005-01-01')) + pd.Timedelta(config.tables.recommand_settings.time_bias)
        self._trade_days = _trade_days[_trade_days <= pd.Timestamp.today() - pd.Timedelta(4, 'h')]
    
    @property
    def portfolio_type(self):
        for i in config.public_keys.recommand_settings.portfolio_types:
            if i in self.table:
                return i
    @property
    def code(self):
        attr = f"{self.portfolio_type}_code"
        return getattr(self, attr)        
    
    @property
    def columns(self) -> Dict:
        """
        ===========================================================================

        Returns the column information for the current table.

        This property dynamically retrieves and formats column metadata,
        including renaming and type mapping.

        Returns
        -------
        Dict
            A dictionary where keys are column names and values are their types.

        ---------------------------------------------------------------------------

        返回当前表的列信息。

        此属性动态检索和格式化列元数据，包括重命名和类型映射。

        返回
        -------
        Dict
            一个字典，其中键是列名，值是其类型。

        ---------------------------------------------------------------------------
        """
        if isinstance(self.columns_information, dict):
            x = merge_dicts(*list(self.columns_information.values()))
        else:
            x = eval(f"jq.get_table_info({self.columns_information})")
            x.iloc[:, 0] = x.iloc[:, 0].replace(config.tables.transform | {'code': self.code})
            x.iloc[:, 2] = x.iloc[:, 2].replace({'date': 'datetime', 'DATE': 'datetime'})
            x = x.set_index(x.columns[0]).iloc[:, [1, 0]].T.to_dict('list')
        return x
    
    def __columns_rename__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ===========================================================================

        Renames columns of the input DataFrame based on predefined mappings.

        This internal method handles column renaming and ensures consistency
        across different data sources, including handling multi-level columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame whose columns need to be renamed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with renamed columns.

        ---------------------------------------------------------------------------

        根据预定义的映射重命名输入 DataFrame 的列。

        此内部方法处理列重命名并确保不同数据源之间的一致性，包括处理多级列。

        参数
        ----------
        df : pd.DataFrame
            需要重命名列的 DataFrame。

        返回
        -------
        pd.DataFrame
            列已重命名的 DataFrame。

        ---------------------------------------------------------------------------
        """
        if isinstance(self.columns_information, dict):
            rename_dic = {i: list(j.keys())[0] for i, j in self.columns_information.items()}
        else:
            rename_dic = config.tables.transform | {'code': self.code}
        df = df.reset_index().rename(rename_dic, axis=1)
        df = df.loc[:, df.columns.isin(list(self.columns.keys()))]
        return df
    
    def __get_data_from_jq_remote__(self, **kwargs: Any) -> pd.DataFrame:
        df = eval(self.commands.format(**kwargs))
        return df    

    def __data_standard__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        # df columns renaed as create_table's columns
        df = self.__columns_rename__(df)
        # add time bias on trade_dt or ann_dt
        for i in [self.ann_dt, self.trade_dt]:
            if i in df.columns:
                df[i] = pd.to_datetime(df[i]) + pd.Timedelta(config.tables.recommand_settings.time_bias)
            if (i not in df.columns) and i in self.columns.keys():
                try:
                    df[i] = pd.to_datetime(kwargs['start_date']) + pd.Timedelta(config.tables.recommand_settings.time_bias)
                except KeyError:
                    pass
        # replace nan and standard codes
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        if self.portfolio_type == 'astock':
            df = df[df[self.code].str.contains(r'^\d', na=False)]
        elif self.portfolio_type == 'afund':
            df = df[df[self.code].isin(self._fund)]
        # standard code code means: 000001.xxxx, check need normalize or not
        if (df[self.code].apply(lambda x: len(x)) != 6 + 1 + 4).any(): 
            df[self.code] = jq.normalize_code(df[self.code].to_list())
        return df

    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        df = self.__get_data_from_jq_remote__(**kwargs)
        func = getattr(self,  f"__data_standard_{self.table}__", self.__data_standard__)
        df = func(df, **kwargs)
        return df

    def __find_max_of_exist_table__(self, columns: str, **kwargs: Any) -> Union[int, float, pd.Timestamp]:
        id_key = None
        if self.table_exist():
            id_key = self.__read__(columns=f'MAX({columns})', show_time=False, **kwargs).iloc[0, 0]
            id_key = None if pd.isnull(id_key) else id_key

        if id_key is None:
            if 'DATE' in self.columns.get(columns, ['None'])[0].upper():
                id_key = pd.to_datetime(getattr(self, 'date_start', config.tables.recommand_settings.date_start))
            else:
                id_key = 0
        return id_key

    def table_exist(self):
        return super().__table_exist__()

    def drop_table(self, **kwargs: Any) -> None:
        parameters = self.__parameters__({'log': True}, kwargs)
        super().__drop_table__(**parameters)

    def create_table(self, **kwargs: Any) -> None:
        parameters = {'columns': self.columns, 'log': True}
        if self.engine_type == 'MySQL':
            keys = (
                self.ann_dt
                if self.trade_dt not in self.columns.keys()
                else self.trade_dt
                )
            partition = None if keys != self.trade_dt else {self.trade_dt: eval(config.tables.recommand_settings.key.partition)}
            parameters = (
                self.__parameters__()
                | {'keys': keys, 'partition': partition}
                | {'columns': self.columns, 'log': True}
                | kwargs
                )
        else:
            parameters = self.__parameters__(parameters, kwargs)
        super().__create_table__(**parameters)


'''
self = main(**config.tables.id_table.astockbalancesheet)

'''
