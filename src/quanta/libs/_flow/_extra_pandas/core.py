# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:40:15 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from quanta.libs._flow.main import __instance__
from quanta.config import settings
col_info = settings('data').public_keys.recommand_settings.key
portfolio_types = settings('data').public_keys.recommand_settings.portfolio_types
config = settings('flow')

@lru_cache(maxsize=8)
def listing(limit=126, portfolio_type='astock'):
    ins = __instance__.get(portfolio_type).listing(limit)
    return ins

@lru_cache(maxsize=8)
def not_st(value=1, portfolio_type='astock'):
    ins = __instance__.get(portfolio_type).not_st(value)
    return ins

@lru_cache(maxsize=8)
def index_members(index_code=None, invert=False):
    if index_code == 'star':
        cols = getattr(__instance__.get('astock'), config.asotck_table)('name').index
        if invert:
            col = [col for col in cols if str(col)[:3] not in config.star_code]
        else:
            col = [col for col in cols if str(col)[:3] in config.star_code]
        x = pd.DataFrame(True, columns=col, index=__instance__.get('trade_days')).loc[config.start_date:]
    else:
        code = config.index_mapping.get(index_code, None)
        if code is None:
            raise ValueError('Undefined index_code: {index_code}')
        x = __instance__.get('aindex').one_to_multi(col_info.astock_code)[code]
    return x

@lru_cache(maxsize=8)
def label(df_obj, code=None, label_df=None, portfolio_type=None):
    if label_df is None:
        portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
        label_df = __instance__.get(portfolio_type)(code)
    else:
        code = label_df.columns.name if code is None else code
    df = pd.concat({code.lower():label_df, 'df':df_obj}, axis=1)
    df = df.stack().set_index(code.lower(), append=True)['df'].unstack(col_info.trade_dt).T
    df.columns = df.columns.swaplevel(-1, 0)
    return df

@lru_cache(maxsize=8)
def expand(df_obj, code=None):
    code = (df_obj.columns.name if code is None else code).lower()
    df_obj = df_obj.stack().to_frame('values')
    for i,j in __instance__.items():
        try:
            df = j(code).stack().to_frame(code).reset_index()
            print(f'get {code} successed in {i}...')
        except:
            pass
    df = pd.merge(df, df_obj.reset_index(), right_on=df_obj.index.names, left_on=[df.columns[0], df.columns[-1]], how='left')
    df = df.set_index(df.columns[:2].to_list()).iloc[:, -1]
    try:
        df = df.unstack()
    except:
        df = df.groupby(df.index.names).sum()
        df = df.unstack()
    return df

def info(df_obj, column, portfolio_type=None):
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    df = __instance__.get(portfolio_type)(column).reindex_like(df_obj)
    return df

def series_info(series_obj, column, portfolio_type=None):
    portfolio_type = series_obj.index.name.split('_')[0] if portfolio_type is None else portfolio_type
    df = __instance__.get(portfolio_type)(column).loc[series_obj.name].reindex(series_obj.index)
    return df

def day_shift(series_obj, n=1, copy=True):
    days = __instance__.get('trade_days')
    day = days.get_loc(series_obj.name) + n
    day = days[day]
    if copy:
        x = series_obj.copy()
        x.name = day
        return x
    else:
        series_obj.name = day
        return series_obj

def parameter_standard(
    parameters: pd.DataFrame,
    sub: float = 0.95,
    **kwargs
) -> pd.DataFrame:
    x = (np.exp(parameters) - sub) / (1 + np.exp(parameters))
    return x

def merge(
    *factors: pd.DataFrame,
    standard: bool = True,
    portfolio_type=None,
    **kwargs
) -> pd.DataFrame:
    if portfolio_type is None:
        portfolio_type = list(set([i.columns.name.split('_')[0] for i in factors]))
        if len(portfolio_type) > 1:
            raise ValueError("factors' portfolio types are not unique...")
        else:
            portfolio_type = portfolio_type[0]
    factors_dict = {i:(j.stats.standard() if standard else j) for i,j in enumerate(factors)}
    ins = __instance__[portfolio_type]
    for i,j in factors_dict.items():
        parameters = ins(config.returns).stats.neutral(fac=j.shift()).params.fac
        parameters = parameter_standard(parameters).rolling(5).mean()
        factors_dict[i] = factors_dict[i].mul(parameters, axis=0)
    factors_merged = pd.concat(factors_dict, axis=1).groupby(factors[0].columns.name, axis=1).mean()
    return factors_merged
