# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:01:26 2026

@author: Porco Rosso
"""

from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import time

__all__ = ['timing_decorator']

def timing_decorator(
    schema: Optional[str] = None, 
    table: Optional[str] = None, 
    show_time: bool = False,
):
    """
    ===========================================================================

    A timer decorator that supports selecting time units.

    Parameters
    ----------
    schema : str, optional
        The database schema name. Defaults to None.
    table : str, optional
        The table name. Defaults to None.
    show_time : bool, optional
        Whether to display the execution time. Defaults to False.

    ---------------------------------------------------------------------------

    支持选择时间单位的计时装饰器。

    参数
    ----------
    schema : str, optional
        数据库模式名称。默认为 None。
    table : str, optional
        表名。默认为 None。
    show_time : bool, optional
        是否显示执行时间。默认为 False。
        
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
