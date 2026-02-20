# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:11:56 2026

@author: Porco Rosso
"""

from functools import lru_cache
from typing import Literal
import pandas as pd

from quanta.libs.db.main import main as db
from quanta.config import settings, login_info
config = settings('data').public_keys
columns_info = config.recommand_settings.key

class main(db, type('public_keys', (), config.recommand_settings.key)):
    time_bias =  pd.Timedelta(config.recommand_settings.time_bias)
    start_date = pd.to_datetime(settings('flow').start_date) + pd.Timedelta(config.recommand_settings.time_bias)

    @classmethod
    @lru_cache(maxsize=1)
    def table_info(cls):
        return cls.__schema_info__()

    @property
    def columns(self):
        x = self.table_info()
        x = x[x['table_name'] == self.table]
        return x['column_name'].to_list()

    @property
    def portfolio_type(self):
        for i in config.recommand_settings.portfolio_types:
            if i in self.table:
                return i

    @property
    def code(self):
        attr = f"{self.portfolio_type}_code"
        return getattr(self, attr)

    @property
    def index_keys(self):
        keys = [i for i in [self.trade_dt, self.ann_dt, self.report_period] if i in self.columns] + [self.code]
        return keys

    @property
    def filter_key(self) -> str:
        return self.index_keys[0]

    def __columns_standard__(self, columns):
        columns = [columns.lower()] if isinstance(columns, str) else [str(i).lower() for i in columns]
        not_have_columns = [i for i in columns if i not in self.columns]
        if not len(not_have_columns):
            return columns
        else:
            raise ValueError(f"Invalid value '{not_have_columns}' for parameter 'columns'. Valid values are: {self.columns}")

    def __read_from_db__(self, returns = False, **kwargs):
        if not hasattr(self, '_internal_data'):
            if self.trade_dt in self.filter_key or self.ann_dt in self.filter_key:
                filter_value = self.start_date if self.filter_key == self.trade_dt else self.start_date + pd.offsets.YearEnd(-4)
                where = f"{self.filter_key} >='{filter_value}'"
            else:
                where = kwargs.get('where', None)
            df = self.__read__(where = where)
            df.columns = pd.CategoricalIndex(df.columns.str.lower())
            df[self.code] = pd.CategoricalIndex(df[self.code])
            df = df.set_index(self.index_keys)
            
            if self.filter_key == self.trade_dt:
                try:
                    columns = [i for i in self.columns if i not in [self.trade_dt, self.ann_dt, self.report_period, self.code]]
                    df = df.unstack()
                    df = {i:df[i] for i in columns}
                except:
                    print(f"WARNING: UNSTANDARD LOADING ON DATA SOURCE <{self.table}>")
            self._internal_data = df
        if returns:
            return self._internal_data

    def __read_from_internal__(self, columns):
        adj = True if isinstance(columns, str) else False
        self.__read_from_db__()
        columns = self.__columns_standard__(columns)
        if self.filter_key == self.trade_dt:
            if not adj:
                df = pd.concat({i:getattr(self, '_internal_data')[i] for i in columns}, axis=1)
                df.columns.names = ['value_key'] + list(df.columns.names)[1:]
            else:
                df = getattr(self, '_internal_data')[columns[0]]
        else:
            df = getattr(self, '_internal_data')[columns]
            if adj:
                df = df.iloc[:, 0]
        return df

    def __call__(
        self,
        columns,
        end = None,
        quarter_adj = False,
        quarter_diff = False,
        shift = 0,
        ** kwargs
    ):
        df = self.__read_from_internal__(columns, **kwargs)
        if end is not None and len(set([self.trade_dt, self.ann_dt]) & self.columns):
            df = df[df.index.get_level_values(self.filter_key) <= end]

        if self.ann_dt in df.index.names:
            df.index = df.index.droplevel(self.ann_dt)
            try:
                df = df.unstack()
                filter_end = pd.Timestamp.today() if end is None else end
                df = df.reindex(pd.date_range(df.index.min(), filter_end, freq='QE', name=df.index.name))
                df = df.sort_index(axis=1).sort_index()
                if quarter_adj:
                    df = self.__finance_quarter_adjust__(df, quarter_adj, quarter_diff)
                if shift:
                    df = self.__finance_shift__(df, shift)
            except:
                pass
        elif self.trade_dt in df.index.names:
            df = df.loc[self.start_date:]
        return df

    def __finance_shift__(self, df: pd.DataFrame, n: int):
        bools = df.iloc[-1].isnull()
        while n > 0 and bools.any():
            n -= 1
            df.loc[:, bools] = df.loc[:, bools].shift()
            bools = df.iloc[-1].isnull()
        return df

    def __finance_quarter_adjust__(
        self,
        df: pd.DataFrame,
        month: int,
        quarter_diff: int
    ) -> pd.DataFrame:
        if self.report_period in df.index.names:
            day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4, 6, 9, 11] else 28)
            index = df.index.get_level_values(self.report_period)
            tmp = df[(index.month == month) & (index.day == day)]
            df = df.diff(quarter_diff)
            df.loc[tmp.index] = tmp
        return df

    def __finance_periods_merge__(
        self,
        df,
        periods,
        quarter_adj=False,
        diff=1,
        drop_zero=False
    ):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.columns=[0]
        df = df.sort_index().reset_index()
        if quarter_adj:
            df = df.sort_values([self.code, self.report_period])
            df['tmp_year'] = df[self.report_period].dt.year
            df['tmp'] = df.groupby(['tmp_year', self.code])[0].diff(diff).fillna(df[0])
            df = df[[self.ann_dt, self.report_period, self.code, 'tmp']].rename({'tmp': 0}, axis=1)

        def next_period(df, shift):
            ne = df.copy()
            ne.columns = ne.columns[:-1].to_list() + [shift*-1]
            ne[self.report_period] = pd.DatetimeIndex(ne[self.report_period]).shift(shift, freq='QE')
            ne = ne.drop(self.ann_dt, errors='ignore', axis=1).set_index([self.report_period, self.code])
            return ne

        merges = [next_period(df, i) for i in range(1, periods)]
        if len(merges):
            merges = pd.concat(merges,axis=1)
            x = pd.merge(df, merges.reset_index(), on=[self.report_period, self.code], how='left')
        else:
            x = df
        x = x.drop_duplicates(subset=[self.ann_dt, self.code], keep='last')
        return x

    @lru_cache(maxsize=8)
    def __finance__(
        self,
        column,
        quarter_adj: bool = False,
        quarter_diff: int = 1,
        shift: int = 0,
        periods: int = 1,
        min_periods = None,
        drop_zero = True,
        day_index = None
    ) -> pd.DataFrame:
        day_index = calendar_days if day_index is None else day_index
        df = self.__read_from_internal__(column)
        if drop_zero:
            df = df[df != 0]
        x = self.__finance_periods_merge__(df, periods, quarter_adj, quarter_diff)
        x = x.set_index(self.ann_dt).sort_index()
        x.index = x.index.astype('datetime64[ns]')
        fill_index = pd.MultiIndex.from_product([day_index, x[self.code].unique()], names=[self.trade_dt, self.code])
        obj = pd.DataFrame(index=fill_index).reset_index(self.code)
        obj = pd.merge_asof(obj, x, by=self.code, left_index=True, right_index=True)

        obj['Q'] = obj.index.to_period('Q')
        obj['Q'] = obj['Q'].dt.year * 4 + obj['Q'].dt.quarter
        obj[self.report_period] = obj[self.report_period].dt.to_period('Q')
        obj[self.report_period] = obj[self.report_period].dt.year * 4 + obj[self.report_period].dt.quarter
        obj['Q'] = obj['Q'] - obj[self.report_period] - 1 - shift
        obj = obj.set_index([self.code, 'Q'], append=True).drop(self.report_period, axis=1).dropna(how='all')
        meta = obj[obj.index.get_level_values('Q') <= 0]
        obj = obj[~obj.index.isin(meta.index)]
        if len(obj):
            while (obj.index.get_level_values('Q') > 0).any():
                obj[obj.index.get_level_values('Q') > 0] = obj[obj.index.get_level_values('Q') > 0].shift(axis=1)
                obj = obj.reset_index('Q')
                obj['Q'] -= 1
                obj = obj.set_index('Q', append=True)
                obj = obj.dropna(how='all')
            obj = pd.concat([meta, obj])
        else:
            obj = meta

        obj = obj.reset_index('Q').drop('Q', axis=1)
        obj.columns.name = self.report_period
        obj = obj.unstack(self.trade_dt)
        obj = obj.swaplevel(self.trade_dt,  self.report_period, axis=1).T
        if len(obj.index.get_level_values(self.report_period).unique()) == 1:
            obj.index = obj.index.get_level_values(self.trade_dt)
        obj = obj.sort_index().sort_index(axis=1)
        if min_periods is not None:
            obj = obj[obj.groupby(self.trade_dt).transform('count') >= min_periods]
        obj = obj[obj.index.get_level_values(0).isin(trade_days)].loc[self.start_date:]
        return obj

calendar_days = pd.date_range(
    start=pd.to_datetime(main.date_start) + main.time_bias,
    freq='d',
    end=pd.Timestamp.today() - pd.Timedelta(4, 'h'))

try:
    import jqdatasdk as jq
    jq.auth(**login_info('account').joinquant)
    trade_days = pd.to_datetime(
        jq.get_trade_days(
            main.date_start,
            pd.Timestamp.today() - pd.Timedelta(4, 'h') - main.time_bias
            )
        ) + main.time_bias
except:
    trade_days = pd.to_datetime(
        sorted(
            main(table='astockeodprices').__read__(columns=main.trade_dt).iloc[:, 0].unique()
            )
        )
