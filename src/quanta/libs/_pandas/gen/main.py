# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:49:44 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Any
from .core import *
MODULE_DIR = __name__.split('.')[-2]

@pd.api.extensions.register_dataframe_accessor(MODULE_DIR)
class main():
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def group(
        self,
        rule = np.linspace(0,1,11).round(2).tolist(),
        pct: bool = True,
        order: bool = False,
        nlevels: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Groups the DataFrame based on specified rules.

        Parameters
        ----------
        rule : Optional[Union[Dict, List]], optional
            The rule for grouping. Can be a dictionary or a list. Defaults to None.
        pct : bool, optional
            Whether to use percentages for grouping. Defaults to True.
        order : bool, optional
            Whether to order the groups. Defaults to False.
        nlevels : Optional[Any], optional
            Number of levels for grouping. Defaults to None.

        Returns
        -------
        pd.DataFrame
            The grouped DataFrame.

        ---------------------------------------------------------------------------

        根据指定规则对 DataFrame 进行分组。

        参数
        ----------
        rule : Optional[Union[Dict, List]], optional
            分组规则。可以是字典或列表。默认为 None。
        pct : bool, optional
            是否使用百分比进行分组。默认为 True。
        order : bool, optional
            是否对组进行排序。默认为 False。
        nlevels : Optional[Any], optional
            分组的级别数。默认为 None。

        返回
        -------
        pd.DataFrame
            分组后的 DataFrame。

        ---------------------------------------------------------------------------
        """
        df: pd.DataFrame = group(self._obj, rule=rule, pct=pct, order=order, nlevels=nlevels)
        return df

    def weight(
        self,
        w_df: Optional[pd.DataFrame] = None,
        fillna: bool = True,
        pct: bool = True,
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Applies weights to the DataFrame.

        Parameters
        ----------
        w_df : Optional[pd.DataFrame], optional
            DataFrame containing weights. Defaults to None.
        fillna : bool, optional
            Whether to fill NaN values. Defaults to True.
        pct : bool, optional
            Whether to use percentages for weights. Defaults to True.

        Returns
        -------
        pd.DataFrame
            The DataFrame with applied weights.

        ---------------------------------------------------------------------------

        对 DataFrame 应用权重。

        参数
        ----------
        w_df : Optional[pd.DataFrame], optional
            包含权重的 DataFrame。默认为 None。
        fillna : bool, optional
            是否填充 NaN 值。默认为 True。
        pct : bool, optional
            是否使用百分比作为权重。默认为 True。

        返回
        -------
        pd.DataFrame
            应用权重后的 DataFrame。

        ---------------------------------------------------------------------------
        """
        return weight(self._obj, w_df=w_df, fillna=fillna, pct=pct)

    def portfolio(
        self,
        returns: pd.DataFrame,
        weight: Optional[pd.DataFrame] = None,
        shift: int = 1,
        roll: int = 1,
        fillna: bool = True
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Constructs a portfolio based on the DataFrame and returns.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame containing returns data.
        weight : Optional[pd.DataFrame], optional
            DataFrame containing weights for portfolio construction. Defaults to None.
        shift : int, optional
            Shift for returns. Defaults to 1.
        roll : int, optional
            Roll for returns. Defaults to 1.
        fillna : bool, optional
            Whether to fill NaN values. Defaults to True.

        Returns
        -------
        pd.DataFrame
            The constructed portfolio DataFrame.

        ---------------------------------------------------------------------------

        根据 DataFrame 和收益构建投资组合。

        参数
        ----------
        returns : pd.DataFrame
            包含收益数据的 DataFrame。
        weight : Optional[pd.DataFrame], optional
            用于投资组合构建的权重 DataFrame。默认为 None。
        shift : int, optional
            收益的位移。默认为 1。
        roll : int, optional
            收益的滚动。默认为 1。
        fillna : bool, optional
            是否填充 NaN 值。默认为 True。

        返回
        -------
        pd.DataFrame
            构建的投资组合 DataFrame。

        ---------------------------------------------------------------------------
        """
        return portfolio(self._obj, returns=returns, weight=weight, shift=shift, roll=roll, fillna=fillna)

    def cut(
        self,
        right: Union[int, float],
        rng_right: Union[int, float] = 0,
        left: Union[int, float] = 0,
        rng_left: Union[int, float] = 0,
        pct: bool = False,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Cuts the DataFrame based on specified boundaries.

        Parameters
        ----------
        right : Union[int, float]
            Right boundary for cutting.
        rng_right : Union[int, float], optional
            Range for the right boundary. Defaults to 0.
        left : Union[int, float], optional
            Left boundary for cutting. Defaults to 0.
        rng_left : Union[int, float], optional
            Range for the left boundary. Defaults to 0.
        pct : bool, optional
            Whether to use percentages for boundaries. Defaults to False.
        ascending : bool, optional
            Whether to sort in ascending order. Defaults to False.

        Returns
        -------
        pd.DataFrame
            The cut DataFrame.

        ---------------------------------------------------------------------------

        根据指定边界切割 DataFrame。

        参数
        ----------
        right : Union[int, float]
            切割的右边界。
        rng_right : Union[int, float], optional
            右边界的范围。默认为 0。
        left : Union[int, float], optional
            切割的左边界。默认为 0。
        rng_left : Union[int, float], optional
            左边界的范围。默认为 0。
        pct : bool, optional
            是否使用百分比作为边界。默认为 False。
        ascending : bool, optional
            是否按升序排序。默认为 False。

        返回
        -------
        pd.DataFrame
            切割后的 DataFrame。

        ---------------------------------------------------------------------------
        """
        return cut(self._obj, left, right, rng_left, rng_right, pct, ascending)

    def part_cut(
        self,
        right: Union[int, float],
        rng_right: Union[int, float] = 0,
        left: Union[int, float] = 0,
        rng_left: Union[int, float] = 0,
        pct: bool = False,
        ascending: bool = False,
        part = 5
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Cuts the DataFrame based on specified boundaries.

        Parameters
        ----------
        right : Union[int, float]
            Right boundary for cutting.
        rng_right : Union[int, float], optional
            Range for the right boundary. Defaults to 0.
        left : Union[int, float], optional
            Left boundary for cutting. Defaults to 0.
        rng_left : Union[int, float], optional
            Range for the left boundary. Defaults to 0.
        pct : bool, optional
            Whether to use percentages for boundaries. Defaults to False.
        ascending : bool, optional
            Whether to sort in ascending order. Defaults to False.

        Returns
        -------
        pd.DataFrame
            The cut DataFrame.

        ---------------------------------------------------------------------------

        根据指定边界切割 DataFrame。

        参数
        ----------
        right : Union[int, float]
            切割的右边界。
        rng_right : Union[int, float], optional
            右边界的范围。默认为 0。
        left : Union[int, float], optional
            切割的左边界。默认为 0。
        rng_left : Union[int, float], optional
            左边界的范围。默认为 0。
        pct : bool, optional
            是否使用百分比作为边界。默认为 False。
        ascending : bool, optional
            是否按升序排序。默认为 False。

        返回
        -------
        pd.DataFrame
            切割后的 DataFrame。

        ---------------------------------------------------------------------------
        """
        return part_cut(self._obj, left, right, rng_left, rng_right, pct, ascending, part)
