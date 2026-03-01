# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 20:00:09 2026

@author: Porco Rosso
"""

from typing import Any, Callable, Dict, List, Optional, Type
from quanta.libs.utils import filter_class_attrs, merge_dicts, timing_decorator


class main:
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
