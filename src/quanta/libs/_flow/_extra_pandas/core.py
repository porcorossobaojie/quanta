# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:40:15 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, make_dataclass
from functools import lru_cache
from typing import Optional, Union, List, Dict, Any, Tuple
from quanta.libs._flow._main import __instance__
from quanta.config import settings
from quanta.libs.utils import dict_to_dataclass

col_info = settings('data').public_keys.recommand_settings.key
portfolio_types = settings('data').public_keys.recommand_settings.portfolio_types
config = settings('flow')


@lru_cache(maxsize=8)
def listing(
    listing_limit: int = 126,
    portfolio_type: str = 'astock'
) -> pd.DataFrame:
    """
    ===========================================================================
    Checks if assets meet the minimum listing duration requirement.

    Parameters
    ----------
    listing_limit : int
        Minimum number of periods since listing. Default is 126.
    portfolio_type : str
        The type of portfolio (e.g., 'astock', 'afund'). Default is 'astock'.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame where True indicates meeting the requirement.
    ---------------------------------------------------------------------------
    检查资产是否满足最小上市时长要求.

    参数
    ----
    listing_limit : int
        自上市以来的最小周期数. 默认为 126.
    portfolio_type : str
        投资组合类型 (例如 'astock', 'afund'). 默认为 'astock'.

    返回
    ----
    pd.DataFrame
        布尔值 DataFrame, True 表示满足要求.
    ---------------------------------------------------------------------------
    """
    ins = __instance__.get(portfolio_type).listing(listing_limit)
    return ins


@lru_cache(maxsize=8)
def not_st(
    value: int = 1,
    portfolio_type: str = 'astock'
) -> pd.DataFrame:
    """
    ===========================================================================
    Filters out assets under Special Treatment (ST) status.

    Parameters
    ----------
    value : int
        The status threshold for ST. Default is 1.
    portfolio_type : str
        The type of portfolio. Default is 'astock'.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame where True indicates non-ST status.
    ---------------------------------------------------------------------------
    过滤掉处于特别处理 (ST) 状态的资产.

    参数
    ----
    value : int
        ST 的状态阈值. 默认为 1.
    portfolio_type : str
        投资组合类型. 默认为 'astock'.

    返回
    ----
    pd.DataFrame
        布尔值 DataFrame, True 表示非 ST 状态.
    ---------------------------------------------------------------------------
    """
    ins = __instance__.get(portfolio_type).not_st(value)
    return ins


@lru_cache(maxsize=8)
def enstatus(
    portfolio_type: str = 'astock',
    periods: Optional[int] = None,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Checks if assets are in a tradable status (e.g., not suspended).

    Parameters
    ----------
    portfolio_type : str
        The type of portfolio. Default is 'astock'.
    periods : Optional[int]
        The rolling window size for status check. Default is 126.
    min_periods : Optional[int]
        Minimum periods required in the window. Default is periods // 2.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame where True indicates tradable status.
    ---------------------------------------------------------------------------
    检查资产是否处于可交易状态 (例如未停牌).

    参数
    ----
    portfolio_type : str
        投资组合类型. 默认为 'astock'.
    periods : Optional[int]
        状态检查的滚动窗口大小. 默认为 126.
    min_periods : Optional[int]
        窗口中所需的最小周期数. 默认为 periods // 2.

    返回
    ----
    pd.DataFrame
        布尔值 DataFrame, True 表示可交易状态.
    ---------------------------------------------------------------------------
    """
    ins = __instance__.get(portfolio_type)(config.status.tradestatus)
    ins = ~ins.astype(bool)
    if periods is not None:
        min_periods = periods // 2 if min_periods is None else min_periods
        ins = ins & (ins.rolling(periods, min_periods=1).sum() > min_periods)
    return ins


@lru_cache(maxsize=8)
def filtered(
    listing_limit: int = 126,
    drop_st: int = 1,
    tradestatus: bool = True,
    portfolio_type: str = 'astock',
    periods: int = 126,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Applies comprehensive filtering including listing duration, ST status,
    and tradability.

    Parameters
    ----------
    listing_limit : int
        Minimum listing duration. Default is 126.
    drop_st : int
        ST status threshold to drop. Default is 1.
    tradestatus : bool
        Whether to filter by tradable status. Default is True.
    portfolio_type : str
        The type of portfolio. Default is 'astock'.
    periods : int
        The rolling window size for status check. Default is 126.
    min_periods : int
        Minimum periods required in the window. Default is periods // 2.

    Returns
    -------
    pd.DataFrame
        Boolean mask of assets meeting all criteria.
    ---------------------------------------------------------------------------
    应用综合过滤, 包括上市时长, ST 状态和可交易性.

    参数
    ----
    listing_limit : int
        最小上市时长. 默认为 126.
    drop_st : int
        要剔除的 ST 状态阈值. 默认为 1.
    tradestatus : bool
        是否按交易状态过滤. 默认为 True.
    portfolio_type : str
        投资组合类型. 默认为 'astock'.
    periods : int
        状态检查的滚动窗口大小. 默认为 126.
    min_periods : int
        窗口中所需的最小周期数. 默认为 periods // 2.

    返回
    ----
    pd.DataFrame
        满足所有条件的资产布尔掩码.
    ---------------------------------------------------------------------------
    """
    dic = {'listing': listing(listing_limit, portfolio_type)}
    if portfolio_type == 'astock':
        if drop_st is not None:
            dic['not_st'] = not_st(drop_st)
        if tradestatus:
            dic['statusable'] = enstatus(periods=periods, min_periods=min_periods)
    count = len(dic)
    dic = pd.concat(dic, axis=1)
    dic = dic.groupby(dic.columns.names[1:], axis=1).sum().astype(int)
    dic = dic >= count
    return dic


@lru_cache(maxsize=8)
def index_members(index_code: Optional[str] = None) -> pd.DataFrame:
    """
    ===========================================================================
    Retrieves a boolean mask of constituents for a given index.

    Parameters
    ----------
    index_code : Optional[str]
        The identifier for the index (e.g., 'star', 'hs300'). Default is None.

    Returns
    -------
    pd.DataFrame
        Boolean mask of index members over time.
    ---------------------------------------------------------------------------
    获取给定指数成份股的布尔掩码.

    参数
    ----
    index_code : Optional[str]
        指数标识符 (例如 'star', 'hs300'). 默认为 None.

    返回
    ----
    pd.DataFrame
        随时间变化的指数成份股布尔掩码.
    ---------------------------------------------------------------------------
    """
    if index_code == 'star':
        cols = getattr(__instance__.get('astock'), config.listing.astock_list.table)(config.listing.astock_list.column).index
        col = [col for col in cols if str(col)[:3] in config.star_code]
        x = pd.DataFrame(True, columns=col, index=__instance__.get('trade_days')).loc[config.start_date:]
    else:
        code = config.index_mapping.get(index_code, None)
        if code is None:
            raise ValueError(f'Undefined index_code: {index_code}')
        x = __instance__.get('aindex').multilize(col_info.astock_code)[code]
    return x


def label(
    code: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    portfolio_type: Optional[str] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Generates labels or indicators for specific categories or asset sets.

    Parameters
    ----------
    code : Optional[str]
        The category/column name to pivot on.
    df : Optional[pd.DataFrame]
        Input data to be labeled. If None, uses multilize on internal data.
    portfolio_type : Optional[str]
        The type of portfolio if df is None.

    Returns
    -------
    pd.DataFrame
        Boolean indicator DataFrame.
    ---------------------------------------------------------------------------
    为特定类别或资产集生成标签或指示器.

    参数
    ----
    code : Optional[str]
        透视所依据的类别/列名.
    df : Optional[pd.DataFrame]
        要标记的输入数据. 如果为 None, 则对内部数据使用 multilize.
    portfolio_type : Optional[str]
        如果 df 为 None 时的投资组合类型.

    返回
    ----
    pd.DataFrame
        布尔指示器 DataFrame.
    ---------------------------------------------------------------------------
    """
    if df is None:
        x = __instance__.get(portfolio_type).multilize(code)
    else:
        x = df.stack().to_frame(code if code is not None else 'other_code')
        x['temp_value'] = 1
        x = x.set_index(x.columns[0], append=True)['temp_value'].unstack(x.index.names[0]).T
        if x.columns.names[0] in [col_info.astock_code, col_info.afund_code]:
            x.columns = x.columns.swaplevel(-1, 0)
        x = x.notnull()
    return x


def expand(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
    level: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Expands a low-level attribute DataFrame to match the multi-level
    index structure of a target DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The attribute data to expand.
    target_df : pd.DataFrame
        The target data with the desired structure.
    level : Optional[Union[int, str]]
        The level name or index to align on.

    Returns
    -------
    pd.DataFrame
        The expanded DataFrame.
    ---------------------------------------------------------------------------
    扩展低级属性 DataFrame 以匹配目标 DataFrame 的多级索引结构.

    参数
    ----
    df : pd.DataFrame
        要扩展的属性数据.
    target_df : pd.DataFrame
        具有所需结构的目标数据.
    level : Optional[Union[int, str]]
        对齐所依据的层级名称或索引.

    返回
    ----
    pd.DataFrame
        扩展后的 DataFrame.
    ---------------------------------------------------------------------------
    """
    level = list(set(df.columns.names) & set(target_df.columns.names))[0] if level is None else level
    x = df.reindex(target_df.columns.get_level_values(level), axis=1)
    x.columns = target_df.columns
    return x


def info(
    df_obj: pd.DataFrame,
    column: str,
    portfolio_type: Optional[str] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Fetches meta-information for each asset in a DataFrame.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input DataFrame indexed by date/asset.
    column : str
        The meta-info column to fetch (e.g., 'industry').
    portfolio_type : Optional[str]
        The portfolio type. Inferred from column names if None.

    Returns
    -------
    pd.DataFrame
        DataFrame of the same shape as df_obj containing meta-info.
    ---------------------------------------------------------------------------
    为 DataFrame 中的每个资产获取元信息.

    参数
    ----
    df_obj : pd.DataFrame
        按日期/资产索引的输入 DataFrame.
    column : str
        要获取的元信息列 (例如 'industry').
    portfolio_type : Optional[str]
        投资组合类型. 如果为 None, 则从列名推断.

    返回
    ----
    pd.DataFrame
        与 df_obj 形状相同且包含元信息的 DataFrame.
    ---------------------------------------------------------------------------
    """
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    df = __instance__.get(portfolio_type)(column).reindex_like(df_obj)
    return df


def series_info(
    series_obj: pd.Series,
    column: str,
    portfolio_type: Optional[str] = None,
    **kwargs
) -> pd.Series:
    """
    ===========================================================================
    Fetches meta-information for assets in a single-period Series.

    Parameters
    ----------
    series_obj : pd.Series
        Series representing assets for a single period.
    column : str
        The meta-info column to fetch.
    portfolio_type : Optional[str]
        The portfolio type.

    Returns
    -------
    pd.Series
        Series containing meta-info.
    ---------------------------------------------------------------------------
    为单周期 Series 中的资产获取元信息.

    参数
    ----
    series_obj : pd.Series
        表示单周期资产的 Series.
    column : str
        要获取的元信息列.
    portfolio_type : Optional[str]
        投资组合类型.

    返回
    ----
    pd.Series
        包含元信息的 Series.
    ---------------------------------------------------------------------------
    """
    kwargs  = {'end': series_obj.name} | kwargs
    portfolio_type = series_obj.index.name.split('_')[0] if portfolio_type is None else portfolio_type
    try:
        df = __instance__.get(portfolio_type)(column).loc[series_obj.name].reindex(series_obj.index)
    except:
        df = __instance__.get(portfolio_type)(column, **kwargs).reindex(series_obj.index, axis=1)
    return df


def day_shift(
    series_obj: pd.Series,
    n: int = 1,
    copy: bool = True
) -> pd.Series:
    """
    ===========================================================================
    Shifts the date name of a Series by a specific number of trading days.

    Parameters
    ----------
    series_obj : pd.Series
        The input Series named by date.
    n : int
        The number of trading days to shift. Default is 1.
    copy : bool
        Whether to return a copy. Default is True.

    Returns
    -------
    pd.Series
        The Series with the shifted date as its name.
    ---------------------------------------------------------------------------
    将 Series 的日期名称平移指定的交易日数.

    参数
    ----
    series_obj : pd.Series
        以日期命名的输入 Series.
    n : int
        平移的交易日数. 默认为 1.
    copy : bool
        是否返回副本. 默认为 True.

    返回
    ----
    pd.Series
        以平移后日期为名称的 Series.
    ---------------------------------------------------------------------------
    """
    if n != 0:
        days = __instance__.get('trade_days')
        day = days.get_loc(series_obj.name) + n
        day = days[day]
    else:
        day = series_obj.name
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
    **kwargs: Any
) -> pd.DataFrame:
    """
    ===========================================================================
    Standardizes regression parameters using a logistic-like transformation.

    Parameters
    ----------
    parameters : pd.DataFrame
        The input regression coefficients.
    sub : float
        The subtraction constant for adjustment. Default is 0.95.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        The standardized parameters.
    ---------------------------------------------------------------------------
    使用类逻辑回归变换对回归参数进行标准化.

    参数
    ----
    parameters : pd.DataFrame
        输入的回归系数.
    sub : float
        用于调整的减法常数. 默认为 0.95.
    **kwargs : Any
        附加关键字参数.

    返回
    ----
    pd.DataFrame
        标准化后的参数.
    ---------------------------------------------------------------------------
    """
    x = (np.exp(parameters) - sub) / (1 + np.exp(parameters))
    return x


def merge(
    *factors: pd.DataFrame,
    standard: bool = True,
    portfolio_type: Optional[str] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    ===========================================================================
    Merges multiple factors into a composite factor using weighted averages
    derived from rolling returns.

    Parameters
    ----------
    *factors : pd.DataFrame
        The factors to be merged.
    standard : bool
        Whether to standardize factors before merging. Default is True.
    portfolio_type : Optional[str]
        The type of portfolio.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        The composite factor.
    ---------------------------------------------------------------------------
    使用从滚动收益率推导出的加权平均值将多个因子合并为综合因子.

    参数
    ----
    *factors : pd.DataFrame
        要合并的因子.
    standard : bool
        合并前是否对因子进行标准化. 默认为 True.
    portfolio_type : Optional[str]
        投资组合类型.
    **kwargs : Any
        附加关键字参数.

    返回
    ----
    pd.DataFrame
        综合因子.
    ---------------------------------------------------------------------------
    """
    if portfolio_type is None:
        types = list(set([i.columns.name.split('_')[0] for i in factors]))
        if len(types) > 1:
            raise ValueError("factors' portfolio types are not unique...")
        else:
            portfolio_type = types[0]
    factors_dict = {i: (j.stats.standard() if standard else j) for i, j in enumerate(factors)}
    ins = __instance__[portfolio_type]
    for i, j in factors_dict.items():
        parameters = ins(config.trade_keys.returns).stats.neutral(fac=j.shift()).params.fac
        parameters = parameter_standard(parameters).rolling(5).mean()
        factors_dict[i] = factors_dict[i].mul(parameters, axis=0)
    factors_merged = pd.concat(factors_dict, axis=1).groupby(factors[0].columns.name, axis=1).mean()
    return factors_merged


def port(
    df_obj: pd.DataFrame,
    ret: Optional[pd.DataFrame] = None,
    listing_limit: int = 126,
    drop_st: int = 1,
    tradestatus: bool = True,
    portfolio_type: Optional[str] = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Generates group returns for a factor DataFrame after applying filters.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input factor values.
    ret : Optional[pd.DataFrame]
        Returns data. If None, fetched from global instance.
    listing_limit : int
        Minimum listing duration. Default is 126.
    drop_st : int
        ST status threshold. Default is 1.
    tradestatus : bool
        Whether to filter by tradable status. Default is True.
    portfolio_type : Optional[str]
        The portfolio type.

    Returns
    -------
    pd.DataFrame
        Group returns over time.
    ---------------------------------------------------------------------------
    在应用过滤后为因子 DataFrame 生成组收益.

    参数
    ----
    df_obj : pd.DataFrame
        输入因子值.
    ret : Optional[pd.DataFrame]
        收益率数据. 如果为 None, 则从全局实例中获取.
    listing_limit : int
        最小上市时长. 默认为 126.
    drop_st : int
        ST 状态阈值. 默认为 1.
    tradestatus : bool
        是否按交易状态过滤. 默认为 True.
    portfolio_type : Optional[str]
        投资组合类型.

    返回
    ----
    pd.DataFrame
        随时间变化的组收益.
    ---------------------------------------------------------------------------
    """
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    if ret is None:
        ret = __instance__[portfolio_type](config.trade_keys.returns)
        filter_df = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(df_obj).fillna(False)
        df_obj = df_obj[filter_df]
    x = df_obj.gen.group().gen.portfolio(ret).loc['2017':]
    return x


def trend(
    df_obj: pd.DataFrame,
    periods: int = 21
) -> pd.DataFrame:
    """
    ===========================================================================
    Calculates the rolling correlation between values and a time index to
    identify trends.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The input data.
    periods : int
        The rolling window size. Default is 21.

    Returns
    -------
    pd.DataFrame
        Rolling correlation values indicating trend strength.
    ---------------------------------------------------------------------------
    计算数值与时间索引之间的滚动相关性以识别趋势.

    参数
    ----
    df_obj : pd.DataFrame
        输入数据.
    periods : int
        滚动窗口大小. 默认为 21.

    返回
    ----
    pd.DataFrame
        指示趋势强度的滚动相关性值.
    ---------------------------------------------------------------------------
    """
    x = pd.DataFrame(np.tile(range(df_obj.shape[0]), (df_obj.shape[1], 1)).T, index=df_obj.index, columns=df_obj.columns)
    x = df_obj.rolling(periods, min_periods=periods//4).corr(x)
    return x


def ic(
    df_obj: pd.DataFrame,
    shift: int = 1,
    listing_limit: int = 126,
    drop_st: int = 1,
    tradestatus: bool = True,
    portfolio_type: Optional[str] = None
) -> pd.Series:
    """
    ===========================================================================
    Calculates Information Coefficient (IC) for a factor.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The factor values.
    shift : int
        Shift of returns for correlation calculation. Default is 1.
    listing_limit : int
        Minimum listing duration. Default is 126.
    drop_st : int
        ST status threshold. Default is 1.
    tradestatus : bool
        Whether to filter by tradable status. Default is True.
    portfolio_type : Optional[str]
        The portfolio type.

    Returns
    -------
    pd.Series
        The daily IC values.
    ---------------------------------------------------------------------------
    计算因子的信息系数 (IC).

    参数
    ----
    df_obj : pd.DataFrame
        因子值.
    shift : int
        和滞后 n 期的 returns 做 IC. 默认为 1.
    listing_limit : int
        最小上市时长. 默认为 126.
    drop_st : int
        ST 状态阈值. 默认为 1.
    tradestatus : bool
        是否按交易状态过滤. 默认为 True.
    portfolio_type : Optional[str]
        投资组合类型.

    返回
    ----
    pd.Series
        每日 IC 值.
    ---------------------------------------------------------------------------
    """
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    filter_df = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(df_obj).fillna(False)
    df_obj = df_obj[filter_df]
    ret = __instance__[portfolio_type](config.trade_keys.returns)
    x = df_obj.shift(shift).corrwith(ret, axis=1)
    return x
        
def ic_predict(
    Series: pd.Series,
    windows: Tuple[int, ...] = (5, 10, 15, 21, 42, 63),
    diff: Tuple[int, ...] = (1,),
    periods: int = 42
) -> pd.DataFrame:
    """
    =======================================================================
    Predicts Information Coefficient (IC) using rolling regression on
    exponentially weighted moving averages and differences.

    Parameters
    ----------
    Series : pd.Series
        The input IC series.
    windows : Tuple[int, ...]
        Lookback windows for weighting. Default is (5, 10, 15, 21, 42, 63).
    diff : Tuple[int, ...]
        Difference orders for trend capture. Default is (1,).
    periods : int
        The rolling window size for the predictive regression. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame containing predicted IC ('p') and R-squared ('r').
    -----------------------------------------------------------------------
    使用指数加权移动平均和差分的滚动回归来预测信息系数 (IC).

    参数
    ----
    Series : pd.Series
        输入的 IC 序列.
    windows : Tuple[int, ...]
        加权的回看窗口. 默认为 (5, 10, 15, 21, 42, 63).
    diff : Tuple[int, ...]
        用于捕获趋势的差分阶数. 默认为 (1,).
    periods : int
        预测回归的滚动窗口大小. 默认为 42.

    返回
    ----
    pd.DataFrame
        包含预测 IC ('p') 和 R 方值 ('r') 的 DataFrame.
    -----------------------------------------------------------------------
    """
    val = {w:pd.tools.array_roll(Series.values.astype('float32')[:, np.newaxis], w) for w in windows}
    val = {i.__str__(): np.einsum('w, twk -> t', pd.tools.halflife(i, i//4), j) for i,j in val.items()}
    for d in diff:
        x = Series.diff(d)
        temp_val = {w:pd.tools.array_roll(x.values.astype('float32')[:, np.newaxis], w) for w in windows}
        temp_val = {f"{i}_{d}": np.einsum('w, twk -> t', pd.tools.halflife(int(i), int(i)//4), j) for i,j in temp_val.items()}
        val = val | temp_val
    val = np.array([np.pad(i, (Series.shape[0]-i.shape[0], 0), mode='constant', constant_values=np.nan) for i in val.values()]).T
    val = np.insert(val, 0, np.ones_like(Series.values), axis=1)
    val = np.insert(val, 0, Series.shift(-1).values.astype('float32'), axis=1)
    val = pd.tools.array_roll(val, periods)
    from quanta.libs._pandas.stats.core import _lstsq
    x = _lstsq(val)[0]
    resid = val[:, :, 0] - np.einsum('tk, twk -> tw', x, val[:, :, 1:])
    r = 1 - resid.var(axis=1) / val[:, :, 0].var(axis=1)
    y_pred =  (x[:-1, :] * val[1:, -1, 1:]).sum(axis=1)
    df = pd.DataFrame(np.array([y_pred, r[:-1]]), columns = Series.index[-y_pred.shape[0]:], index=['p', 'r']).T
    return df
    
def ir(
    df_obj: Union[pd.DataFrame, pd.Series],
    periods: int = 126,
    listing_limit: int = 126,
    drop_st: int = 1,
    tradestatus: bool = True,
    portfolio_type: Optional[str] = None
) -> pd.Series:
    """
    ===========================================================================
    Calculates Information Ratio (IR) based on IC values.

    Parameters
    ----------
    df_obj : Union[pd.DataFrame, pd.Series]
        Either factor values (DataFrame) or pre-calculated IC (Series).
    periods : int
        The rolling window size. Default is 126.
    listing_limit : int
        Listing filter limit. Default is 126.
    drop_st : int
        ST filter threshold. Default is 1.
    tradestatus : bool
        Tradability filter indicator. Default is True.
    portfolio_type : Optional[str]
        The portfolio type.

    Returns
    -------
    pd.Series
        The rolling IR values.
    ---------------------------------------------------------------------------
    基于 IC 值计算信息比率 (IR).

    参数
    ----
    df_obj : Union[pd.DataFrame, pd.Series]
        因子值 (DataFrame) 或预先计算的 IC (Series).
    periods : int
        滚动窗口大小. 默认为 126.
    listing_limit : int
        上市过滤限制. 默认为 126.
    drop_st : int
        ST 过滤阈值. 默认为 1.
    tradestatus : bool
        可交易性过滤指示. 默认为 True.
    portfolio_type : Optional[str]
        投资组合类型.

    返回
    ----
    pd.Series
        滚动 IR 值.
    ---------------------------------------------------------------------------
    """
    if isinstance(df_obj, pd.DataFrame):
        df_obj = ic(df_obj, listing_limit, drop_st, tradestatus, portfolio_type)
    x = df_obj.rolling(periods, min_periods=periods // 4)
    ir_val = x.mean() / x.std()
    return ir_val

def corr(
    df_obj: pd.DataFrame,
    other_obj: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    =======================================================================
    Calculates cross-sectional correlation between factors or with a shift.

    Parameters
    ----------
    df_obj : pd.DataFrame
        The primary factor DataFrame.
    other_obj : Optional[pd.DataFrame]
        The factor to correlate with. If None, uses df_obj.shift().

    Returns
    -------
    pd.Series
        Descriptive statistics of the cross-sectional correlation.
    -----------------------------------------------------------------------
    计算因子之间或带有平移的横截面相关性.

    参数
    ----
    df_obj : pd.DataFrame
        主因子 DataFrame.
    other_obj : Optional[pd.DataFrame]
        要与之关联的因子. 如果为 None, 则使用 df_obj.shift().

    返回
    ----
    pd.Series
        横截面相关性的描述性统计.
    -----------------------------------------------------------------------
    """
    if other_obj is None:
        return df_obj.corrwith(df_obj.shift(), axis=1).describe()
    else:
        return df_obj.corrwith(other_obj, axis=1).describe()

class test:
    """
    ===========================================================================
    High-level backtesting engine for rapid strategy validation.

    This class provides a streamlined interface for calculating theoretical
    and real-world portfolio returns, considering trade constraints like
    price limits and costs.
    ---------------------------------------------------------------------------
    用于快速策略验证的高级回测引擎.

    此类提供了一个简化的接口, 用于计算理论和现实世界的投资组合收益率, 同时考虑
    价格限制和成本等交易约束.
    ---------------------------------------------------------------------------
    """
    trade_price = config.trade_keys.avgprice_adj
    settle_price = config.trade_keys.close_adj
    compare_price = config.trade_keys.avgprice
    high = config.trade_keys.high_limit
    low = config.trade_keys.low_limit
    limit = 0.01
    cost = 0.00075

    def __init__(
        self,
        df: pd.DataFrame,
        shift: int = 1
    ):
        """Initializes the backtest instance | 初始化回测实例"""
        df = df.astype('float').shift(shift)
        df = df[df > 0].dropna(how='all', axis=1).dropna(how='all', axis=0)
        df = df.div(df.sum(axis=1), axis=0)
        self._internal_data = df.fillna(0)

    @lru_cache(maxsize=1)
    def _low_limit(self, limit: float) -> pd.DataFrame:
        """Checks for instruments hitting the low limit | 检查触及跌停板的标的"""
        x = 1 - self._internal_data.f.info(self.low) / self._internal_data.f.info(self.compare_price) < limit
        return x

    @property
    def low_limit(self) -> pd.DataFrame:
        """Returns low limit indicator DataFrame | 返回跌停指示 DataFrame"""
        return self._low_limit(self.limit)

    @lru_cache(maxsize=1)
    def _high_limit(self, limit: float) -> pd.DataFrame:
        """Checks for instruments hitting the high limit | 检查触及涨停板的标的"""
        x = self._internal_data.f.info(self.high) / self._internal_data.f.info(self.compare_price) - 1 < limit
        return x

    @property
    def high_limit(self) -> pd.DataFrame:
        """Returns high limit indicator DataFrame | 返回涨停指示 DataFrame"""
        return self._high_limit(self.limit)

    @property
    @lru_cache(maxsize=1)
    def entrade(self) -> pd.DataFrame:
        """
        =======================================================================
        Calculates tradable positions considering price limit restrictions.

        Returns
        -------
        pd.DataFrame
            The actual tradable position weights.
        -----------------------------------------------------------------------
        考虑价格限制约束计算可交易持仓.

        返回
        ----
        pd.DataFrame
            实际可交易的持仓权重.
        -----------------------------------------------------------------------
        """
        df = self._internal_data
        values = df.values
        high = self.high_limit.values
        low = self.low_limit.values
        lst = np.zeros_like(values)
        lst[0] = np.where(~high[0], values[0], 0)

        for i in range(1, values.shape[0]):
            diff = values[i] - lst[i-1]
            lst[i] = np.where(
                ((diff > 0) & (~high[i])) | ((diff < 0) & (~low[i])),
                values[i],
                lst[i-1]
            )
        x = pd.DataFrame(lst, index=df.index, columns=df.columns)
        return x

    @property
    @lru_cache(maxsize=1)
    def trade_difference(self) -> pd.DataFrame:
        """Returns the difference between target and actual trades | 返回目标与实际交易之间的差异"""
        x = self._internal_data - self.entrade
        x = x[x != 0].dropna(how='all', axis=1).dropna(how='all', axis=0)
        return x

    @property
    @lru_cache(maxsize=1)
    def unbuy(self) -> pd.DataFrame:
        """Returns assets that couldn't be bought due to limits | 返回因限制而无法买入的资产"""
        x = self.trade_difference
        x = x[x > 0].dropna(how='all', axis=1).dropna(how='all', axis=0)
        return x

    @property
    @lru_cache(maxsize=1)
    def unsell(self) -> pd.DataFrame:
        """Returns assets that couldn't be sold due to limits | 返回因限制而无法卖出的资产"""
        x = self.trade_difference
        x = x[x < 0].dropna(how='all', axis=1).dropna(how='all', axis=0)
        return x

    def action_returns(self, df: pd.DataFrame) -> Any:
        """Calculates returns specifically for buy and sell actions | 专门计算买入和卖出行为的收益率"""
        buy_actions = df[(df.shift() == 0) & df > 0].notnull()
        buy_ret = (buy_actions.f.info(self.trade_price) / buy_actions.f.info(self.settle_price))[buy_actions] - 1
        sell_actions =  df[(df.shift(-1) == 0) & df > 0].notnull()
        sell_ret = ( sell_actions.f.info(self.trade_price) / sell_actions.f.info(self.settle_price).shift())[sell_actions] - 1
        return type('returns', (), {'buy':buy_ret, 'sell':sell_ret})

    def real_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates actual returns for a given position DataFrame | 计算给定持仓 DataFrame 的实际收益率"""
        action_returns = self.action_returns(df)
        real_returns = action_returns.buy.fillna(df.f.info(config.trade_keys.returns))
        real_returns = real_returns[df > 0].fillna(action_returns.sell)
        return real_returns

    def summary(self, rebalance: bool = True) -> Any:
        """
        =======================================================================
        Generates a comprehensive summary of backtest performance.

        Parameters
        ----------
        rebalance : bool
            Whether to re-weight positions to 1.0 each period. Default is True.

        Returns
        -------
        Any
            A structured object containing returns, turnover, and order details.
        -----------------------------------------------------------------------
        生成回测绩效的综合摘要.

        参数
        ----
        rebalance : bool
            是否每期将持仓权重重新调整为 1.0. 默认为 True.

        返回
        ----
        Any
            包含收益率, 换手率和订单详情的结构化对象.
        -----------------------------------------------------------------------
        """
        thoery_returns = (self._internal_data * self._internal_data.f.info(config.trade_keys.returns)).sum(axis=1)
        thoery_turnover = self._internal_data.diff().fillna(self._internal_data).abs().sum(axis=1)
        thoery_cost = thoery_turnover * self.cost

        entrade = self.entrade
        if rebalance:
            entrade = entrade.div(entrade.sum(axis=1), axis=0)
        real_returns = (entrade * self.real_returns(entrade)).sum(axis=1)
        real_turnover = entrade.diff().fillna(entrade).abs().sum(axis=1)
        real_cost = real_turnover * self.cost

        dic = {
            'returns':{
                'thoery': {
                    'pure': thoery_returns,
                    'cost':thoery_cost,
                    'final': thoery_returns - thoery_cost
                },
                'real': {
                    'pure': real_returns,
                    'cost': real_cost,
                    'final': real_returns - real_cost,
                    'detail': self.real_returns(self.entrade)
                },
                'action': self.action_returns(self.entrade)
            },
            'turnover':{
                'thoery': thoery_turnover,
                'real': real_turnover
            },
            'order':{
                'thoery': self._internal_data,
                'real': self.entrade,
                'difference': self.trade_difference,
                'unbuy': self.unbuy,
                'unsell': self.unsell
            }
        }
        return dict_to_dataclass(dic)

def concept(df_obj, label, label_df=None, expand=True, how='sum', w=None, portfolio_type=None):
    x = df_obj.f.label(label, label_df, portfolio_type)
    if w is not None:
        w = w.f.label(label, label_df, portfolio_type)
        x = (x * w).groupby(x.columns.names[0], axis=1).apply(how) / w.groupby(w.columns.names[0], axis=1).apply(how)
    else:
        x = x.groupby(x.columns.names[0], axis=1).apply(how)
    if expand:
        x = x.f.expand()
        x = x.groupby(x.columns.names[1], axis=1).mean()
    return x
