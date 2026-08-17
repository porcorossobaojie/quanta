# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 20:00:09 2026

@author: Porco Rosso
"""
import pandas as pd
from pathlib import Path

from pathlib import Path
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Type
from ....libs.utils import filter_class_attrs, merge_dicts, timing_decorator

class main:
    """
    ===========================================================================
    Base class for database engines, providing timing decorators, MRO
    parameter merging, and instance update protocols.
    ---------------------------------------------------------------------------
    数据库引擎基类, 提供计时装饰器, MRO 参数合并和实例更新协议.
    ---------------------------------------------------------------------------
    """

    @classmethod
    def __timing_decorator__(
        cls,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        show_time: bool = False
    ) -> Callable[..., Any]:
        """
        =======================================================================
        Returns a timing decorator configured for a specific data source.

        Parameters
        ----------
        schema : str, optional
            The database schema name, by default None.
        table : str, optional
            The table name, by default None.
        show_time : bool, optional
            Whether to display execution time, by default False.

        Returns
        -------
        Callable
            A configured timing decorator instance.
        -----------------------------------------------------------------------
        返回为特定数据源配置的计时装饰器.

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
            一个已配置的计时装饰器实例.
        -----------------------------------------------------------------------
        """
        return timing_decorator(schema, table, show_time)

    @classmethod
    def __get_all_parents_dict__(cls) -> List[Type[Any]]:
        """Internal protocol method for MRO analysis | 内部协议方法: MRO 分析"""
        return [
            parent for parent in cls.mro()
            if (parent is not object and hasattr(cls, 'mro'))
        ][::-1]

    def __parameters__(self, *args: Dict[str, Any]) -> Dict[str, Any]:
        """
        =======================================================================
        Merges class attributes, parent attributes, and provided arguments.

        Parameters
        ----------
        *args : Dict[str, Any]
            Additional parameters to merge.

        Returns
        -------
        Dict[str, Any]
            The final merged parameter dictionary.
        -----------------------------------------------------------------------
        合并类属性, 父类属性以及提供的参数.

        参数
        ----
        *args : Dict[str, Any]
            要合并的额外参数.

        返回
        ----
        Dict[str, Any]
            最终合并后的参数字典.
        -----------------------------------------------------------------------
        """
        all_sources = [
            *[filter_class_attrs(i) for i in self.__get_all_parents_dict__()],
            filter_class_attrs(self),
            *args
        ]
        res = merge_dicts(*all_sources)

        final_params = {}
        for k, v in res.items():
            if isinstance(v, property):
                final_params[k] = getattr(self, k)
            else:
                final_params[k] = v

        return final_params

    def __call__(self, replace: bool = False, **kwargs: Any) -> Optional['main']:
        """
        =======================================================================
        Updates instance parameters or returns a new instance with updates.

        Parameters
        ----------
        replace : bool, optional
            If True, returns a new instance; otherwise updates in-place,
            by default False.
        **kwargs : Any
            New parameters to apply.

        Returns
        -------
        Optional[main]
            A new instance if replace is True, else None.
        -----------------------------------------------------------------------
        更新实例参数或返回带有更新的新实例.

        参数
        ----
        replace : bool, optional
            如果为 True, 返回新实例; 否则进行就地更新, 默认为 False.
        **kwargs : Any
            要应用的新参数.

        返回
        ----
        Optional[main]
            若 replace 为 True 则返回新实例, 否则返回 None.
        -----------------------------------------------------------------------
        """
        parameters = self.__parameters__(kwargs)
        if replace:
            return self.__class__(**parameters)
        else:
            [setattr(self, i, j) for i, j in parameters.items()]
        
    @classmethod
    def __read_parquet__(cls, path, file_name=None):
        """
        =======================================================================
        Reads a parquet file into a DataFrame via DuckDB.

        Parameters
        ----------
        path : str
            The directory or file path of the parquet data.
        file_name : str, optional
            The parquet file name; when provided, path is treated as a
            directory. Default is None.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from the parquet file.
        -----------------------------------------------------------------------
        通过 DuckDB 将 parquet 文件读取为 DataFrame.

        参数
        ----
        path : str
            parquet 数据的目录或文件路径.
        file_name : str, optional
            parquet 文件名; 提供时 path 被视为目录. 默认 None.

        返回
        ----
        pd.DataFrame
            从 parquet 文件加载的 DataFrame.
        -----------------------------------------------------------------------
        """
        if file_name is not None:
            if file_name.split('.')[-1] != 'parquet':
                file_name = f"{file_name}.parquet"
            x = pd.read_parquet(f"{path}/{file_name}")
        else:
            x = pd.read_parquet(f"{path}")
        return x

    @classmethod
    def __write_parquet__(cls, df, path, file_name=None, log=True):
        """
        =======================================================================
        Writes a DataFrame to a parquet file under the given path.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to persist.
        path : str
            The destination directory for the parquet file.
        file_name : str, optional
            The parquet file name; defaults to 'data_0' when None.
        log : bool
            Whether to print a write summary. Default is True.
        -----------------------------------------------------------------------
        将 DataFrame 写入指定路径下的 parquet 文件.

        参数
        ----
        df : pd.DataFrame
            需要持久化的 DataFrame.
        path : str
            parquet 文件的输出目录.
        file_name : str, optional
            parquet 文件名; 为 None 时默认 'data_0'.
        log : bool
            是否打印写入摘要. 默认 True.
        -----------------------------------------------------------------------
        """
        file_name = 'data_0' if file_name is None else file_name
        if file_name.split('.')[-1] != 'parquet':
            file_name = f"{file_name}.parquet"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(f"{path}/{file_name}")
        if log:
            print(f"Written DataFrame to <{path}.{file_name}>: {df.shape[0]} records.")
