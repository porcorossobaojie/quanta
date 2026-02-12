# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 20:00:09 2026

@author: Porco Rosso
"""

from typing import Any, Callable, Dict, List, Optional, Type
from quanta.libs.utils import filter_class_attrs, merge_dicts, timing_decorator

class main():
    @classmethod
    def __timing_decorator__(
        cls,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        show_time: bool = False
    ) -> Callable[..., Any]:
        return timing_decorator(schema, table, show_time)

    @classmethod
    def __get_all_parents_dict__(cls) -> List[Type[Any]]:
        return [
            parent for parent in cls.mro()
            if (parent is not object and hasattr(cls, 'mro'))
        ][::-1]

    def __parameters__(self, *args: Dict[str, Any]) -> Dict[str, Any]:
        all_sources = [
            *[filter_class_attrs(i) for i in self.__get_all_parents_dict__()],
            filter_class_attrs(self),
            *args
        ]
        return merge_dicts(*all_sources)

    def __call__(self, replace: bool = False, **kwargs: Any) -> Optional['main']:
        parameters = self.__parameters__(kwargs)
        if replace:
            return self.__class__(**parameters)
        else:
            [setattr(self, i, j) for i, j in parameters.items()]
