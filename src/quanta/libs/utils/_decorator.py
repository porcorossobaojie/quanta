# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:01:26 2026

@author: Porco Rosso
"""

from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import time

__all__ = ['timing_decorator', 'doc_inherit']


def doc_inherit(source_func: Any):
    """
    ===========================================================================
    A decorator that copies the docstring from a source function to the
    decorated method.

    Parameters
    ----------
    source_func : Any
        The function from which to inherit the docstring.

    Returns
    -------
    Callable
        The decorated method with the inherited docstring.
    ---------------------------------------------------------------------------
    从源函数复制文档字符串到装饰方法的装饰器.

    参数
    ----
    source_func : Any
        从中继承文档字符串的函数.

    返回
    ----
    Callable
        具有继承文档字符串的装饰方法.
    ---------------------------------------------------------------------------
    """
    def decorator(target_func):
        target_func.__doc__ = source_func.__doc__
        return target_func
    return decorator


def timing_decorator(
    schema: Optional[str] = None,
    table: Optional[str] = None,
    show_time: bool = False,
):
    """
    ===========================================================================
    A timer decorator that supports logging execution time for data
    source operations.

    Parameters
    ----------
    schema : str, optional
        The database schema name, by default None.
    table : str, optional
        The table name, by default None.
    show_time : bool, optional
        Whether to display the execution time, by default False.

    Returns
    -------
    Callable
        The decorated function with timing logic.
    ---------------------------------------------------------------------------
    支持记录数据源操作执行时间的计时装饰器.

    参数
    ----
    schema : str, optional
        数据库模式名称, 默认为 None.
    table : str, optional
        表名, 默认为 None.
    show_time : bool, optional
        是否显示执行时间, 默认为 False.

    返回
    ----
    Callable
        带有计时逻辑的装饰函数.
    ---------------------------------------------------------------------------
    """
    def decorator(func):
        @wraps(func)
        def wrapper(
            *args: Any,
            **kwargs: Any
        ):
            if show_time:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                if execution_time >= 0.1:
                    unit = 's'
                    print(
                        f"Data Source <{schema}.{table}> executed in {execution_time:.3f}{unit}"
                    )
                elif execution_time * 1e3 >= 0.01:
                    execution_time = execution_time * 1e3
                    unit = 'ms'
                    print(
                        f"Data Source <{schema}.{table}> executed in {execution_time:.3f}{unit}"
                    )
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
