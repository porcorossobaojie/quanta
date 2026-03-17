# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:40:15 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Optional, Union, List, Dict, Any, Tuple
from quanta.libs._flow._main import __instance__
from quanta.config import settings

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
    periods: int = 126,
    min_periods:int = None
) -> pd.DataFrame:
    """
    ===========================================================================
    Checks if assets are in a tradable status (e.g., not suspended).

    Parameters
    ----------
    portfolio_type : str
        The type of portfolio. Default is 'astock'.
    periods : int
        The rolling window size for status check. Default is 126.
    min_periods : int
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
    periods : int
        状态检查的滚动窗口大小. 默认为 126.
    min_periods : int
        窗口中所需的最小周期数. 默认为 periods // 2.

    返回
    ----
    pd.DataFrame
        布尔值 DataFrame, True 表示可交易状态.
    ---------------------------------------------------------------------------
    """
    min_periods = periods // 2 if min_periods is None else min_periods
    ins = __instance__.get(portfolio_type)(config.status.tradestatus)
    ins = ~ins.astype(bool)
    if periods is not None:
        ins = ins & (ins.rolling(periods, min_periods=1).sum() > min_periods)
    return ins


@lru_cache(maxsize=8)
def filtered(
    listing_limit: int = 126,
    drop_st: int = 1,
    tradestatus: bool = True,
    portfolio_type: str = 'astock',
    periods: int = 126,
    min_periods:int = None
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
        parameters = ins(config.returns).stats.neutral(fac=j.shift()).params.fac
        parameters = parameter_standard(parameters).rolling(5).mean()
        factors_dict[i] = factors_dict[i].mul(parameters, axis=0)
    factors_merged = pd.concat(factors_dict, axis=1).groupby(factors[0].columns.name, axis=1).mean()
    return factors_merged


def port(
    df_obj: pd.DataFrame,
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
    filter_df = filtered(listing_limit, drop_st, tradestatus, portfolio_type).reindex_like(df_obj).fillna(False)
    df_obj = df_obj[filter_df]
    ret = __instance__[portfolio_type](config.trade_keys.returns)
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
    x = df_obj.shift().corrwith(ret, axis=1)
    return x


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


def qtest(
    df_obj: pd.DataFrame,
    high: Optional[str] = None,
    low: Optional[str] = None,
    avgprice: Optional[str] = None,
    trade_price: Optional[str] = None,
    settle_price: Optional[str] = None,
    limit: float = 0.01,
    trade_cost: float = 0.0015,
    portfolio_type: Optional[str] = None
) -> Any:
    """
    ===========================================================================
    Executes a quantitative backtest (quick test) on a target portfolio
    weight DataFrame.

    Parameters
    ----------
    df_obj : pd.DataFrame
        Target portfolio weights.
    high : Optional[str]
        Column name for high price limits.
    low : Optional[str]
        Column name for low price limits.
    avgprice : Optional[str]
        Column name for average prices.
    trade_price : Optional[str]
        Column name for execution prices.
    settle_price : Optional[str]
        Column name for settlement prices.
    limit : float
        Threshold for price-limit-induced untradability. Default is 0.01.
    trade_cost : float
        Transaction cost percentage. Default is 0.0015.
    portfolio_type : Optional[str]
        The portfolio type.

    Returns
    -------
    back_test
        A custom object containing backtest results (order, trade, settle).
    ---------------------------------------------------------------------------
    对目标组合权重 DataFrame 执行量化回测 (quick test).

    参数
    ----
    df_obj : pd.DataFrame
        目标组合权重.
    high : Optional[str]
        最高价限制的列名.
    low : Optional[str]
        最低价限制的列名.
    avgprice : Optional[str]
        平均价格的列名.
    trade_price : Optional[str]
        执行价格的列名.
    settle_price : Optional[str]
        结算价格的列名.
    limit : float
        由价格限制引起的不交易性阈值. 默认为 0.01.
    trade_cost : float
        交易成本百分比. 默认为 0.0015.
    portfolio_type : Optional[str]
        投资组合类型.

    返回
    ----
    back_test
        包含回测结果 (订单, 交易, 结算) 的自定义对象.
    ---------------------------------------------------------------------------
    """
    portfolio_type = df_obj.columns.name.split('_')[0] if portfolio_type is None else portfolio_type
    high_key = config.trade_keys.high_limit if high is None else high
    low_key = config.trade_keys.low_limit if low is None else low
    avg_key = config.trade_keys.avgprice if avgprice is None else avgprice
    trade_key = config.trade_keys.avgprice_adj if trade_price is None else trade_price
    settle_key = config.trade_keys.close_adj if settle_price is None else settle_price
    meta_data = df_obj.copy()

    df_obj = df_obj.replace(0, np.nan).dropna(how='all').div(df_obj.sum(axis=1, min_count=1), axis=0).fillna(0)
    ins = __instance__[portfolio_type]
    buyable = ((1 - ins(avg_key) / ins(high_key)) >= limit).reindex_like(df_obj).fillna(False)
    sellable = ((1 - ins(low_key) / ins(avg_key)) >= limit).reindex_like(df_obj).fillna(False)
    trade_vals = ins(trade_key).reindex_like(df_obj)
    settle_vals = ins(settle_key).reindex_like(df_obj)
    trader_status = enstatus(portfolio_type).reindex_like(df_obj).fillna(False)

    values = {
        'buyable': buyable.values,
        'sellable': sellable.values,
        'trade': trade_vals.values,
        'settle': settle_vals.values,
        'order': df_obj.values,
        'trader': trader_status.values
    }

    start = np.where(
        values['buyable'][0] & values['trader'][0], values['order'][0], 0
    )
    portfolio_trade = [
        start / start.sum() * (1 - trade_cost)
    ] # 交易的资产, 1为本金,扣除交易费用
        
    portfolio_change = [
        np.nan_to_num(
            portfolio_trade[0] / values['trade'][0],
            nan=0
        )
    ] # 交易的股数
    portfolio_hold = [
        portfolio_change[0]
    ] # 期末持有的股数
    portfolio_hold = [portfolio_change[0]]
    portfolio_settle = [
        np.nan_to_num(
            portfolio_hold[0] * values['settle'][0],
            nan=0
        )
    ] # 期末持有的资产
    portfolio_different = [np.where(~values['buyable'][0] | ~values['trader'][0], values['order'][0], 0)] # 未能交易的资产
    for i in range(1, df_obj.shape[0]):
        diff = (values['order'][i] - portfolio_settle[-1] / portfolio_settle[-1].sum()) * portfolio_settle[-1].sum() # 要交易的资产
        meta_diff = diff.copy()
        diff = np.nan_to_num(
            diff / values['settle'][i-1],
            nan=0
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
        portfolio_trade.append(np.nan_to_num(diff * values['trade'][i], nan=0))
        portfolio_hold.append(portfolio_hold[-1] + diff)
        portfolio_settle.append(np.nan_to_num(portfolio_hold[-1] * values['settle'][i], nan=0))
        portfolio_different.append(different)

    portfolio_trade = pd.DataFrame(portfolio_trade, index=df_obj.index, columns=df_obj.columns)
    portfolio_change = pd.DataFrame(portfolio_change, index=df_obj.index, columns=df_obj.columns)
    portfolio_hold = pd.DataFrame(portfolio_hold, index=df_obj.index, columns=df_obj.columns)
    portfolio_settle = pd.DataFrame(portfolio_settle, index=df_obj.index, columns=df_obj.columns)
    portfolio_different = pd.DataFrame(portfolio_different, index=df_obj.index, columns=df_obj.columns)

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
