# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:49:40 2026

@author: admin
"""
import numpy as np
import pandas as pd
from numba import njit, prange
from itertools import product
from typing import Optional, Dict, List, Union

from quanta.libs._pandas.tools.core import fillna as fillna_func


@njit(parallel=True, cache=True, nopython=True)
def fast_rank(data_2d, rule):
    result = np.full(data_2d.shape, np.nan)
    for i in prange(data_2d.shape[0]):
        mask = ~np.isnan(data_2d[i])
        slice_data = data_2d[i][mask]
        if len(slice_data):
            count = slice_data.shape[0]
            slice_data = slice_data.argsort().argsort() + 1
            slice_data = slice_data / count
            slice_data = np.searchsorted(rule, slice_data, side='left')
            slice_data = np.fmax(1, np.fmin(len(rule) - 1, slice_data))
            result[i, mask] = slice_data
    return result


def group(
    df: pd.DataFrame,
    rule: Union[Dict, List],
    order: bool = True,
) -> pd.DataFrame:
    is_multi = bool(df.columns.nlevels - 1)
    rule = {i:np.array(j) for i,j in rule.items()} if isinstance(rule, dict) else np.array(rule)

    if not is_multi:
        df = pd.DataFrame(fast_rank(df.values, rule), index=df.index, columns=df.columns).astype('Int64')
    else:
        if isinstance(rule, np.ndarray):
            rule = {i: rule for i in df.columns.get_level_values(0).unique()}
        keys = list(rule.keys())
        x = df.sort_index(axis=1)
        cols = x.columns.get_level_values(-1).value_counts() == len(keys)
        cols = cols[cols].index
        x = x.loc[:, x.columns.get_level_values(-1).isin(cols)]
        arrays = np.full((x.index.shape[0], cols.shape[0], len(keys)), np.nan)
        if order:
            arrays[:, :, 0] = fast_rank(x[keys[0]].values, rule[keys[0]])
            ruled = [range(1, len(rule[keys[0]]))]
            for i in range(1, len(keys)):
                flat_values = arrays[:, :, :i]
                unique_keys = np.array(list(product(*ruled)))
                result = (flat_values[:, :, np.newaxis, :] == unique_keys[np.newaxis, np.newaxis, :, :])
                result = result.all(axis=-1)
                result = np.where(result, x[keys[i]].values[:, :, np.newaxis], np.nan)
                result = result.transpose(2, 0, 1).reshape(-1, result.shape[1])
                result = fast_rank(result, rule[keys[i]])
                result = result.reshape(len(unique_keys), -1, result.shape[-1])
                result = np.nansum(result, axis=0)
                arrays[:, :, i] = result
                ruled.append(range(1, len(rule[keys[i]])))
        else:
            for i in range(len(keys)):
                arrays[:, :, i] =fast_rank(x[keys[i]].values, rule[keys[i]])
        df = pd.DataFrame(arrays.reshape(-1, len(keys))).fillna(-1).astype(int).astype(str)
        df = pd.Series(df.values.tolist()).str.join('_')
        df = pd.DataFrame(df.values.reshape(x.shape[0], -1), index=x.index, columns=cols)
        df = df.stack()
        df = df[~df.str.contains('-1',  regex=False)].unstack()
    return df
