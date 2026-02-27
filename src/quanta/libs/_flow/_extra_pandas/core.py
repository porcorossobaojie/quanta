# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:40:15 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from quanta.libs._flow._main import __instance__
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
def statusable(portfolio_type='astock'):
    ins = __instance__.get(portfolio_type)(config.status.tradestatus)
    ins = ~ins.astype(bool)
    return ins

@lru_cache(maxsize=8)
def filtered(listing_limit=126, drop_st=1, tradestatus=True, portfolio_type='astock'):
    dic = {'listing': __instance__[portfolio_type].listing(listing_limit, portfolio_type)}
    if portfolio_type == 'astock':
        if not_st is not None:
            dic['not_st'] = not_st(drop_st)
        if tradestatus:
            dic['statusable'] = statusable()
    count = len(dic)
    dic = pd.concat(dic, axis=1)
    dic = dic.groupby(dic.columns.names[1:], axis=1).sum().astype(int)
    dic = dic >= count
    return dic

@lru_cache(maxsize=8)
def index_members(index_code=None):
    if index_code == 'star':
        cols = getattr(__instance__.get('astock'), config.listing.astock_list.table)(config.listing.astock_list.column).index
        col = [col for col in cols if str(col)[:3] in config.star_code]
        x = pd.DataFrame(True, columns=col, index=__instance__.get('trade_days')).loc[config.start_date:]
    else:
        code = config.index_mapping.get(index_code, None)
        if code is None:
            raise ValueError('Undefined index_code: {index_code}')
        x = __instance__.get('aindex').multilize(col_info.astock_code)[code]
    return x

def label(code=None, df=None, portfolio_type=None):
    if df is None:
        x = __instance__.get(portfolio_type).multilize(code)
    else:
        x = df.stack().to_frame(code if code is not None else 'other_code')
        x['temp_value'] = 1
        x = x.set_index(x.columns[0], append=True)['temp_value'].unstack(x.index.names[0]).T
        if x.columns.names[0] in [col_info.astock_code, col_info.afund_code]:
            x.columns = x.columns.swaplevel(-1,0)
        x = x.notnull()
    return x

def expand(df, target_df, level=None):
    level = list(set(df.columns.names) & set(target_df.columns.names))[0] if level is None else level
    x = df.reindex(target_df.columns.get_level_values(level), axis=1)
    x.columns = target_df.columns
    return x

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

def port(df_obj, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    filter_df = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(df_obj).fillna(False)
    df_obj = df_obj[filter_df]
    ret = __instance__[portfolio_type](config.trade_keys.returns)
    x = df_obj.gen.group().gen.portfolio(ret).loc['2017:']
    return x

def trend(df_obj, periods=21):
    x = pd.DataFrame(np.tile(range(df_obj.shape[0]), (df_obj.shape[1], 1)).T, index=df_obj.index, columns=df_obj.columns)
    x = df_obj.rolling(periods, min_periods=periods//4).corr(x)
    return x

def ic(df_obj, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    filter_df = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(df_obj).fillna(False)
    df_obj = df_obj[filter_df]
    ret = __instance__[portfolio_type](config.trade_keys.returns)
    x = df_obj.shift().corrwith(ret, axis=1)
    return x

def ir(df_obj, periods=126, listing_limit=126, drop_st=1, tradestatus=True, portfolio_type=None):
    if isinstance(df_obj, pd.DataFrame):
        df_obj = ic(df_obj, listing_limit, drop_st, tradestatus, portfolio_type)
    x = df_obj.rolling(periods, min_periods=periods // 4)
    ir = x.mean() / x.std()
    return ir

def qtest(
    df_obj,
    high=None,
    low=None,
    avgprice=None,
    trade_price=None,
    settle_price=None,
    limit=0.01,
    trade_cost = 0.0015,
    portfolio_type=None
):
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    high_key = config.trade_keys.high_limit if high is None else high
    low_key = config.trade_keys.low_limit if low is None else low
    avg_key = config.trade_keys.avgprice if avgprice is None else avgprice
    trade_key = config.trade_keys.avgprice_adj if trade_price is None else trade_price
    settle_key = config.trade_keys.close_adj if settle_price is None else settle_price
    meta_data = df_obj.copy()

    df_obj = df_obj.replace(0, np.nan).dropna(how='all').div(df_obj.sum(axis=1, min_count=1), axis=0).fillna(0)
    ins = __instance__[portfolio_type]
    buyable = ((1 -  ins(avg_key) / ins(high_key)) >= limit).reindex_like(df_obj).fillna(False)
    sellable = ((1 - ins(low_key) / ins(avg_key)) >= limit).reindex_like(df_obj).fillna(False)
    trade = ins(trade_key).reindex_like(df_obj)
    settle = ins(settle_key).reindex_like(df_obj)
    trader = statusable(portfolio_type).reindex_like(df_obj).fillna(False)

    values = {
        'buyable':buyable.values,
        'sellable':sellable.values,
        'trade':trade.values,
        'settle':settle.values,
        'order': df_obj.values,
        'trader':trader.values
    }

    start = np.where(
        values['buyable'][0] & values['trader'][0], values['order'][0], 0
    )
    portfolio_trade = [
        start / start.sum()  * (1 - trade_cost)
    ] # 交易的资产, 1为本金,扣除交易费用
    portfolio_change = [
        np.nan_to_num(
            portfolio_trade[0] / values['trade'][0],
            nan = 0
        )
    ] # 交易的股数
    portfolio_hold = [portfolio_change[0]] # 期末持有的股数
    portfolio_settle = [
        np.nan_to_num(
            portfolio_hold[0] * values['settle'][0],
            nan = 0
        )
    ] # 期末持有的资产
    portfolio_different = [np.where(~values['buyable'][0] | ~values['trader'][0], values['order'][0], 0)] # 未能交易的资产
    for i in range(1, df_obj.shape[0]):
        diff = (values['order'][i] - portfolio_settle[-1] / portfolio_settle[-1].sum()) * portfolio_settle[-1].sum() # 要交易的资产
        meta_diff = diff.copy()
        diff = np.nan_to_num(
            diff / values['settle'][i-1],
            nan = 0
        ) # 转化成股份
        diff = np.where(
            (((diff < 0) & values['sellable'][i]) | ((diff > 0) & values['buyable'][i])) & values['trader'][i],
            diff,
            0
        )
        different = np.where((meta_diff != 0) & (diff == 0), meta_diff, 0)
        sells = np.where(diff < 0, diff, 0)
        buy = np.where(diff > 0, diff, 0)
        buy = -np.nansum(sells * values['trade'][i]) / np.nansum(buy * values['trade'][i]) * buy * (1 - trade_cost)
        diff = sells + buy

        portfolio_change.append(diff)
        portfolio_trade.append(np.nan_to_num(diff * values['trade'][i], nan = 0))
        portfolio_hold.append(portfolio_hold[-1] + diff)
        portfolio_settle.append(np.nan_to_num(portfolio_hold[-1] * values['settle'][i], nan = 0))
        portfolio_different.append(different)

    portfolio_trade = pd.DataFrame(portfolio_trade, index=df_obj.index, columns = df_obj.columns)
    portfolio_change = pd.DataFrame(portfolio_change, index=df_obj.index, columns = df_obj.columns)
    portfolio_hold = pd.DataFrame(portfolio_hold, index=df_obj.index, columns = df_obj.columns)
    portfolio_settle = pd.DataFrame(portfolio_settle, index=df_obj.index, columns = df_obj.columns)
    portfolio_different = pd.DataFrame(portfolio_different, index=df_obj.index, columns = df_obj.columns)
    settle_bool = portfolio_settle > 0
    order_bool = df_obj > 0
    different = (settle_bool.astype(int) - order_bool.astype(int))
    class back_test:
        class order:
            data = meta_data
            weight = df_obj
            limit = portfolio_different
        class trade:
            assets = portfolio_trade
            shares = portfolio_change
        class settle:
            assets = portfolio_settle
            shares = portfolio_hold
    return back_test




