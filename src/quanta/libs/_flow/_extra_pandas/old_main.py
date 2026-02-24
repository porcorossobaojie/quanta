# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:35:28 2025

@author: Porco Rosso

"""
from functools import lru_cache
from libs.__flow__.__init__ import stock, index, trade_days, fund
from libs.__flow__.config import FACTORIZE, COLUMNS_INFO
import pandas as pd
import numpy as np

PCT_CHANGE = 's_dq_pctchange'

DIC = {
       'fund':fund,
       'index':index,
       'stock':stock
       }


def _be_list(df_obj=None, limit=None, inplace=True):
    x = stock.be_list(limit=limit if limit is not None else FACTORIZE.on_list_limit)
    if df_obj is not None:
        x = x.reindex_like(df_obj).fillna(False)
    if inplace and df_obj is not None:
        return df_obj[x]
    else:
        return x

def _not_st(df_obj=None, limit=0, inplace=True):
    x = stock.is_st() <= limit
    if df_obj is not None:
        x = x.reindex_like(df_obj).fillna(False)
    if inplace and df_obj is not None:
        return df_obj[x]
    else:
        return x

def members(df_obj=None, name=None, inplace=True):
    if name == 'star':
        x = _be_list(df_obj=None, limit=1)
        x = x.loc[:, [i[:3] not in FACTORIZE.star_info for i in x.columns]]
    elif name is not None:
        name = FACTORIZE.index_mapping[name]
        x = index.index_member()[name].notnull()
    else:
        x = _be_list(df_obj=None, limit=1)
    if df_obj is not None:
        x = x.reindex_like(df_obj).fillna(False)
    if inplace and df_obj is not None:
        return df_obj[x]
    else:
        return x

def label(df_obj, key=None, label_df=None, field='stock'):
    dic = {'fund':fund.traced_index,
           'index':index,
           'stock':stock}
    field = dic.get(field, stock)
    if key is None:
        key = 'S_INFO_LABEL'
    else:
        key, label_obj = key.upper(), field(key)

    if COLUMNS_INFO.code not in label_obj.index.names:
        x = pd.concat({key:label_obj, '__FACTOR__': df_obj}, axis=1).stack(COLUMNS_INFO.code)
    else:
        df_obj = df_obj.stack(COLUMNS_INFO.code)
        x = pd.merge(label_obj.iloc[:, -1].to_frame(key), df_obj.to_frame('__FACTOR__'), left_index=True, right_index=True, how='left')
    x = x.set_index(key, append=True)['__FACTOR__']
    x = x.unstack([key, COLUMNS_INFO.code])
    return x

def expand(df_obj, label=None, field='stock'):
    dic = {'fund':fund.traced_index,
           'index':index,
           'stock':stock}
    field = dic.get(field, stock)
    label = df_obj.columns.name if label is None else label.upper()
    label_df = field(label)
    duplicate = True
    if label_df.index.nlevels == 1:
        duplicate = False
        label_df = label_df.stack().to_frame(label)
    label_df = label_df.reset_index()
    df = df_obj.stack().to_frame('__LABEL_VALUE__').reset_index()
    df = pd.merge(label_df, df, left_on=[label_df.columns[0], label], right_on=[df_obj.index.name, label], how='left').drop(label, axis=1)
    df = df.set_index(label_df.columns[:2].to_list())
    if duplicate:
        print(f'Index contains duplicate entries, label name: <{label}>')
        df = df.groupby(df.index.names).sum()
    df = df.loc[:, '__LABEL_VALUE__'].unstack()
    return df

def info(df_obj, key, field='stock'):
    func = DIC.get(field, stock)
    x = func(key).reindex_like(df_obj)
    return x

def sinfo(series, key, field='stock'):
    func = DIC.get(field, stock)
    x = func(key)
    if x.index.name == COLUMNS_INFO.trade_dt:
        x = x.loc[series.name].reindex(series.index)
    else:
        x = stock(key, end=series.name)
    return x

def shift(series, n=1, copy=True):
    days = trade_days()
    day = days.get_loc(series.name) + n
    day = days[day]
    if copy:
        x = series.copy()
        x.name = day
        return x
    else:
        series.name = day

def parameter_standard(
    parameters: pd.DataFrame,
    sub: float = 0.95
) -> pd.DataFrame:
    x = (np.exp(parameters) - sub) / (1 + np.exp(parameters))
    return x

def expose(y, expose_values, *xs, limit=0.05, max_iters=2):
    _xs = {i:j for i,j in enumerate(xs)} if isinstance(xs, (list, tuple)) else {0:xs}
    def expose_1dim(y, x, v):
        y = y.stats.neutral(fac=x).resid
        y_std = y.std(axis=1)
        beta = (y_std * v) / (x.std(axis=1) * (1 - v ** 2))
        hat = y + x.mul(beta, axis=0)
        return hat
    if len(_xs.keys()) == 1:
        df = expose_1dim(y, _xs[0], expose_values)
    else:
        df = y.stats.neutral(**{i.__str__():j for i,j in _xs.items()}).resid
        itered = 0
        len_xs = len(_xs.keys())
        while (itered < max_iters) and len_xs:
            for i,j in _xs.items():
                df = expose_1dim(df, j, expose_values[i])
            check = {i:df.corrwith(j, axis=1).mean() for i,j in enumerate(xs)}
            check = [i for i,j in check.items() if np.abs(j - expose_values[i]) > limit]
            _xs = {i:xs[i] for i in check}
            len_xs = len(_xs.keys())
            itered += 1
            print(f"optmize itered: {itered}, needed optmize count: {len_xs}")
        for i,j in enumerate(xs):
            value = round(df.corrwith(j, axis=1).mean(), 4)
            print(f"x{i}: expose_value: {expose_values[i]} optmized: {value}")
    return df

def _tradeable(df_obj=None, inplace=True):
    x = stock('S_DQ_TRADESTATUS')
    if df_obj is not None:
        x = ~x.reindex_like(df_obj).fillna(1).astype(bool)
    if inplace and df_obj is not None:
        return df_obj[x]
    else:
        return x

def reversal(df_obj, how='mean', field='stock'):
    func = DIC.get(field, stock)
    x = func(PCT_CHANGE)
    if how == 'max':
        pct = x.rolling(3).max()
    elif how == 'mean':
        pct = x.rolling(3).mean()
    return df_obj.stats.neutral(pct=pct).resid

def merge(
    *factors: pd.DataFrame,
    standard: bool = True
) -> pd.DataFrame:
    factors_dict = {i:(j.stats.standard() if standard else j) for i,j in enumerate(factors)}
    for i,j in factors_dict.items():
        parameters = stock(PCT_CHANGE).stats.neutral(fac=j.shift()).params.fac
        parameters = parameter_standard(parameters).rolling(5).mean()
        factors_dict[i] = factors_dict[i].mul(parameters, axis=0)
    factors_merged = pd.concat(factors_dict, axis=1).groupby(COLUMNS_INFO.code, axis=1).mean()
    return factors_merged

def port(df_obj, name=None, not_st=True, be_list=True, field='stock'):
    func = DIC.get(field, stock)
    ret = func(PCT_CHANGE)
    if field == 'stock':
        df_obj = df_obj[recommand_filter(not_st=not_st, be_list=be_list).reindex_like(df_obj).fillna(False)]
        df_obj = members(df_obj, name=name)
    x = df_obj.build.group()
    x = x.build.portfolio(ret).loc['2017':]
    return x

def ic(df_obj, not_st=True, be_list=True, field='stock'):
    func = DIC.get(field, stock)
    ret = func(PCT_CHANGE)
    if field == 'stock':
        df_obj =df_obj[recommand_filter(not_st=not_st, be_list=be_list).reindex_like(ret).fillna(False)]
    ic = df_obj.shift().corrwith(ret, axis=1)
    return ic

def icir(sereis, periods=252):
    ir = sereis.rolling(periods, min_periods=periods // 4)
    ir = ir.mean() / ir.std()
    return ir

def trend(df_obj, periods=21):
    x = pd.DataFrame(np.tile(range(df_obj.shape[0]), (df_obj.shape[1], 1)).T, index=df_obj.index, columns=df_obj.columns)
    x = df_obj.rolling(periods, min_periods=periods//4).corr(x)
    return x

def detrend(df_obj, indust='S_SWL1_CODE', top_quantile=0.9, bottom_quantile=0.1, not_st=True, be_list=True):
    if not_st:
        df_obj = _not_st(df_obj)
    if be_list:
        df_obj = _be_list(df_obj)
    df_obj = label(df_obj, indust)
    top = df_obj.groupby(indust.upper(), axis=1).quantile(top_quantile)
    bot = df_obj.groupby(indust.upper(), axis=1).quantile(bottom_quantile)
    df = (top - bot).mean(axis=1)
    return df

def crowd(df_obj, periods=21, not_st=True, be_list=True):
    turnover = (stock('s_dq_freeturnover')/100).rolling(periods, min_periods=periods//4).mean()
    ret = stock(PCT_CHANGE)
    ret_std = ret.rolling(periods, min_periods=periods//4).std()
    df_obj = _tradeable(df_obj)
    if not_st:
        df_obj = _not_st(df_obj)
    if be_list:
        df_obj = _be_list(df_obj)
    ret = ret.reindex_like(df_obj)[df_obj.notnull()]
    bench = ret.mean(axis=1)
    bench = pd.DataFrame(np.tile(bench.values, (df_obj.shape[1], 1)).T, index=df_obj.index, columns=df_obj.columns)
    beta = ret.rolling(periods, min_periods=periods // 4).cov(bench) / bench.rolling(periods, min_periods=periods // 4).var().replace(0, np.nan)

    obj = pd.concat(
        {
         'factor': df_obj.rank(axis=1, pct=True).build.group([0,0.3,0.7,1.0]),
         'turnover': turnover,
         'std': ret_std,
         'beta':beta
            },
        axis=1).stack()
    obj = obj.groupby([COLUMNS_INFO.trade_dt, 'factor']).mean().unstack()
    obj = obj.groupby(obj.columns.get_level_values(0), axis=1).apply(lambda x: x[x.columns.max()] - x[x.columns.min()]).mean(axis=1)
    return obj

@lru_cache(maxsize=16)
def recommand_filter(be_list=True, not_st=True, tradeable=True, forcast=True):
    df = _be_list() if be_list else _be_list(limit=1)
    if not_st:
        df = _not_st(df)
    if tradeable:
        df = _tradeable(df)
    if forcast:
        x = pd.db.read(table ='ashareforcast')
        index = [COLUMNS_INFO.ann_dt, COLUMNS_INFO.code]
        x = x.sort_values(index).drop_duplicates(index, keep='last')
        x = x[x['TYPE_ID'] >= 305007].set_index(index)['TYPE_ID'].unstack()
        x = x.sort_index().reindex(pd.date_range(x.index.min(), x.index.max(), freq='d')).shift().ffill(limit=4)
        x = x.reindex_like(df)
        df = df[x.isnull()]
    df = df.notnull()
    return df

def compare(df, date, lst1, count, lst2=None):
    x = df.loc[date]
    x1 = x.loc[lst1].sort_values(ascending=False).iloc[:count]
    if lst2 is None:
        x2 = x.drop(x1.index).sort_values(ascending=False).iloc[:count]
    else:
        x2 = x.loc[lst2].sort_values(ascending=False).iloc[:count]
    x1 = x1.to_frame('returns').reset_index()
    x2 = x2.to_frame('returns').reset_index()
    x2.index = list(range(x2.shape[0]-1, -1, -1))
    x = pd.concat({'hold': x1, 'expert': x2}, axis=1)
    return x

def signal_merge(*signal_obj):
    factors = {}
    ws = {}
    for i,j in enumerate(signal_obj):
        x = j()[j.w >= 0]
        ws[i] = j.w
        x = x.mul(((j.w - 1).tools.log() + 1), axis=0)
        factors[i] = x
    factors = pd.concat(factors, axis=1).groupby(COLUMNS_INFO.code, axis=1).sum(min_count=1)
    ws = pd.concat(ws, axis=1)
    factors = factors.div(((ws - 1).tools.log() + 1).sum(axis=1), axis=0)
    x1 = signal_obj
    class signal:
        signal_obj = x1
        factor = factors
        signal = ws
    return signal

def prem_cut(*signal_obj, count=50, prem=0.01, max_count_limit=None, max_turnover_limit=0.1, member=None):
    factor_group = signal_merge(*signal_obj)
    factors = factor_group.factor
    ws = factor_group.signal

    factors = members(factors, name=member).dropna(how='all', axis=1).dropna(how='all', axis=0)
    factors = factors[factors.notnull().sum(axis=1) >= count * 2]
    changes = min(int(count * max_turnover_limit), count if max_count_limit is None else max_count_limit)

    lst = [factors.iloc[0].sort_values(ascending=False)[:count]]
    for i in factors.index[1:]:
        x = compare(factors, i, lst[-1].index, count)
        diff = x['expert']['returns'] - x['hold']['returns'].fillna(0)
        diff = diff[diff >= prem].iloc[-1*changes:]
        stocks = x['expert'][COLUMNS_INFO.code].where(x.index.isin(diff.index), x['hold'][COLUMNS_INFO.code])
        lst.append(factors.loc[i, stocks])
    lst = pd.concat(lst, axis=1).T
    x1 = signal_obj
    class prem():
        strategy = lst
        signal = ws
        factor = factors
        signal_obj = x1
    return prem

def high_limit(df, grouped=False):
    x = df.copy()
    if not grouped:
        x = x.build.group()
        x = x[x == x.stack().max()].shift()
    x = x.notnull()
    high_limit = (stock('s_dq_avgprice') / stock('s_dq_high_limit') >= 0.99)
    high_limit= _tradeable(high_limit).reindex_like(x).fillna(False)
    start_at = pd.DataFrame(
        (~x.shift(fill_value=False)).values & x.values & high_limit.values,
        index=x.index,
        columns=x.columns
    )
    step = pd.DataFrame(
        x.values & start_at.shift(fill_value=False).values & high_limit.values,
        index=x.index,
        columns=x.columns
    )
    start_at = pd.DataFrame(
        start_at | step,
        index=x.index,
        columns=x.columns
    )
    while step.values.sum() > 0:
        step = pd.DataFrame(
            x.notnull().values & step.shift(fill_value=False).values & high_limit.values,
            index = x.index,
            columns = x.columns
        )
        start_at = pd.DataFrame(
            start_at | step,
            index=x.index,
            columns=x.columns
        )
    return start_at

def high_limit_different(df, grouped=False):
    x = df.copy()[recommand_filter().reindex_like(df).fillna(False)]
    if not grouped:
        x = x.build.group()
        x = x[x == x.stack().max()].shift()
    thoe = x.notnull()
    ret = stock('s_dq_pctchange')
    actu = high_limit(x, True)
    dic = {
        'thoery': ret[thoe].mean(axis=1).loc['2017':],
        'actural': ret[thoe & ~actu].mean(axis=1).loc['2017':]
        }
    dic = pd.concat(dic, axis=1)
    dic['diff'] = dic['actural'] - dic['thoery']
    return dic
