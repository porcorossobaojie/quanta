# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:58:29 2026

@author: Porco Rosso
"""

from pprint import pformat
import pandas as pd
import keyword
from typing import Dict, List, Any, Type, Tuple
from dataclasses import make_dataclass, replace

__all__ = [
    'filter_class_attrs', 'filter_parents_class_attrs', 'merge_dicts',
    'flatten_list', 'DictBundle', 'dict_to_dataclass'
]

def filter_class_attrs(class_object: Type[Any]) -> Dict[str, Any]:
    """
    ===========================================================================
    Extracts attributes and their values from a class object, excluding
    dunder methods.

    Parameters
    ----------
    class_object : Type[Any]
        The class object from which to extract attributes.

    Returns
    -------
    Dict[str, Any]
        A dictionary where keys are attribute names and values are their
        corresponding values.
    ---------------------------------------------------------------------------
    从类对象中提取属性及其值, 排除双下划线方法.

    参数
    class_object : Type[Any]
        要从中提取属性的类对象.

    返回
    ----
    Dict[str, Any]
        一个字典, 其中键是属性名称, 值是其对应的值.
    ---------------------------------------------------------------------------
    """
    def check_double_underscore(s: str) -> bool:
        """Check if string is a dunder method | 检查字符串是否为双下划线方法"""
        return s.startswith('__') and s.endswith('__') and len(s) > 4

    dic = {
        i: j for i, j in class_object.__dict__.items()
        if not check_double_underscore(i)
    }
    return dic

def filter_parents_class_attrs(class_object: Type[Any]) -> Dict[str, Any]:
    """
    ===========================================================================
    Extracts attributes from a class and all its parent classes (MRO),
    excluding dunder methods.

    Parameters
    ----------
    class_object : Type[Any]
        The class object from which to extract inherited attributes.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing merged attributes from the entire
        inheritance hierarchy.
    ---------------------------------------------------------------------------
    从类及其所有父类(MRO)中提取属性, 排除双下划线方法.

    参数
    ----
    class_object : Type[Any]
        要从中提取继承属性的类对象.

    返回
    ----
    Dict[str, Any]
        包含从整个继承层次结构合并属性的字典.
    ---------------------------------------------------------------------------
    """
    def check_double_underscore(s: str) -> bool:
        """Check if string is a dunder method | 检查字符串是否为双下划线方法"""
        return s.startswith('__') and s.endswith('__') and len(s) > 4
    all_attrs = {}
    for cls in reversed(class_object.__mro__):  # 反向遍历 MRO, 先处理基类
        # 忽略 object 类, 因为它包含大量不相关的内置属性
        if cls is object:
            continue

        # 获取当前类 (cls) 的直接属性
        current_cls_attrs = {}
        for name, value in cls.__dict__.items():
            if not check_double_underscore(name):
                current_cls_attrs[name] = value

        # 将当前类的属性合并到最终字典中
        # 由于我们是反向遍历 MRO, 先加入基类的属性, 然后子类的同名属性会覆盖它们
        all_attrs.update(current_cls_attrs)

    return all_attrs

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    ===========================================================================
    Merges multiple dictionaries into one, with later dictionaries
    overriding earlier ones for common keys.

    Parameters
    ----------
    *dicts : Dict[str, Any]
        Variable number of dictionaries to merge.

    Returns
    -------
    Dict[str, Any]
        A new dictionary containing the merged key-value pairs.
    ---------------------------------------------------------------------------
    将多个字典合并为一个, 后面字典的键值会覆盖前面字典的键值.

    参数
    ----
    *dicts : Dict[str, Any]
        要合并的字典的可变数量参数.

    返回
    ----
    Dict[str, Any]
        包含合并后的键值对的新字典.
    ---------------------------------------------------------------------------
    """
    all_keys = set().union(*dicts)

    def get_non_none_value(
        key: str
    ) -> Any:
        """Get the last non-None value for a key | 获取键的最后一个非 None 值"""
        return next(
            (d[key] for d in reversed(dicts) if key in d and d[key] is not None),
            None
        )

    return {
        key: get_non_none_value(key) for key in all_keys
    }

def flatten_list(lst: List[Any]) -> List[Any]:
    """
    ===========================================================================
    Flattens a nested list into a single-level list.

    Parameters
    ----------
    lst : List[Any]
        The nested list to flatten.

    Returns
    -------
    List[Any]
        A new list containing all elements from the nested list in a single
        level.
    ---------------------------------------------------------------------------
    将嵌套列表展平为单层列表.

    参数
    ----
    lst : List[Any]
        要展平的嵌套列表.

    返回
    ----
    List[Any]
        一个新列表, 包含嵌套列表中所有元素, 且为单层.
    ---------------------------------------------------------------------------
    """
    return sum(
        (flatten_list(i) if isinstance(i, list) else [i] for i in lst), []
    )
class DictBundle:
    """
    ===========================================================================
    Factory for dict-backed dynamic dataclass instances with field-wise
    pandas dispatch.

    ``build`` turns a nested dictionary into an instance whose fields are the
    dictionary keys. Field values that are instances of ``pandas_types``
    accept any pandas API through attribute lookup: ``obj.rank(axis=1)`` or
    ``obj.T`` applies the same call to every such field and returns a new
    instance of the same dynamic class, leaving the receiver untouched.
    Values of any other type are never operated on and are carried into the
    returned object unchanged.

    Generated classes are cached per field signature, so repeated builds of
    the same structure share one class and ``isinstance`` works across calls.

    Parameters
    ----------
    dic : Dict[str, Any]
        The dictionary to convert. Nested dictionaries become nested objects.
    name : str
        The name of the generated class. Default is "Root".
    pandas_types : Tuple[type, ...]
        Field types eligible for dispatch. Default is DataFrame and Series.

    Returns
    -------
    Any
        An instance of the generated class with ``to_dict``, structure aware
        ``repr`` and pandas dispatch.

    Notes
    -----
    Keys must be valid python identifiers, otherwise ValueError is raised.
    Nested dictionaries become classes named after their dotted path, such as
    ``Neutral.resid``, so nested names stay readable and unambiguous.
    ``to_dict`` returns a shallow nested dictionary: values of ``pandas_types``
    are kept by reference, nested generated objects are unwrapped recursively.
    Instances cannot be pickled, because the class lives outside any module
    namespace. Any other type of value stays untouched.
    ---------------------------------------------------------------------------
    把嵌套字典转换为支持字段级 pandas 分发的动态 dataclass 实例的工厂.

    ``build`` 把嵌套字典转换为字段即字典键的实例. 值为 ``pandas_types`` 实例的
    字段接受任意 pandas API 的属性访问: ``obj.rank(axis=1)`` 或 ``obj.T`` 会把
    同一个调用作用到每个这类字段上, 并返回同一动态类的新实例, 原对象保持不变.
    其他类型的值不被操作, 原样带入返回的对象.

    生成的类按字段签名缓存, 同一结构的多次构建共享同一个类, 因此跨调用
    ``isinstance`` 判断可用.

    参数
    ----
    dic : Dict[str, Any]
        要转换的字典. 嵌套字典会转换为嵌套对象.
    name : str
        生成类的名称. 默认为 "Root".
    pandas_types : Tuple[type, ...]
        参与分发的字段类型. 默认为 DataFrame 与 Series.

    返回
    ----
    Any
        具备 ``to_dict``, 结构感知 ``repr`` 与 pandas 分发的生成类实例.

    说明
    ----
    键必须是合法的 python 标识符, 否则抛出 ValueError. 嵌套字典转换成的类按点
    分路径命名, 例如 ``Neutral.resid``, 嵌套类名因此可读且无歧义. ``to_dict``
    返回浅层嵌套字典: ``pandas_types`` 的实例按引用保留, 嵌套生成对象递归展开.
    实例不能被 pickle, 因为类不在任何模块命名空间内. 其他类型的值保持原样.
    ---------------------------------------------------------------------------
    """
    flag = '__dict_bundle__'
    """Marker attribute set on generated classes | 生成类上的标记属性"""

    _cache: Dict[Any, type] = {}
    """Generated class per field signature | 按字段签名缓存的生成类"""

    @classmethod
    def build(
        cls,
        dic: Dict[str, Any],
        name: str = "Root",
        pandas_types: Tuple[type, ...] = (pd.DataFrame, pd.Series)
    ) -> Any:
        """Builds one instance from a nested dict | 由嵌套字典构建一个实例"""
        if not isinstance(dic, dict):
            raise TypeError(
                f"DictBundle.build expects a dict, got {type(dic).__name__}"
            )

        field_values: Dict[str, Any] = {}
        for key, value in dic.items():
            cls._check_field_name(key)
            if isinstance(value, dict):
                field_values[key] = cls.build(
                    value, name=f"{name}.{key}", pandas_types=pandas_types
                )
            else:
                field_values[key] = value

        signature = (
            name,
            tuple((key, type(value)) for key, value in field_values.items()),
            pandas_types,
        )
        dynamic_cls = cls._cache.get(signature)
        if dynamic_cls is None:
            dynamic_cls = make_dataclass(
                name,
                [(key, Any) for key in field_values],
                namespace={
                    '__repr__': cls._repr,
                    '__getattr__': cls._dispatch,
                    'to_dict': cls._to_dict,
                    '_to_dict': cls._to_dict,
                    cls.flag: True,
                    '_pandas_types': pandas_types,
                },
                eq=False,
            )
            cls._cache[signature] = dynamic_cls
        return dynamic_cls(**field_values)

    @classmethod
    def build_legacy(
        cls,
        dic: Dict[str, Any],
        name: str = "Root"
    ) -> Any:
        """Builds one instance with default pandas types | 用默认 pandas 类型构建实例"""
        return cls.build(dic, name=name)

    @staticmethod
    def _check_field_name(key: Any) -> None:
        """Validates that a key can become a field | 校验键能否作为字段名"""
        if (
            not isinstance(key, str)
            or not key.isidentifier()
            or keyword.iskeyword(key)
        ):
            raise ValueError(
                f"Field name {key!r} is not a valid python identifier; "
                "rename the key before building the object"
            )

    @staticmethod
    def _plain_value(value: Any, pandas_types: Tuple[type, ...]) -> Any:
        """Unwraps generated objects into plain containers | 递归还原为普通容器"""
        if isinstance(value, pandas_types):
            return value
        if getattr(type(value), DictBundle.flag, False):
            return value.to_dict()
        if isinstance(value, dict):
            return {
                k: DictBundle._plain_value(v, pandas_types)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [DictBundle._plain_value(v, pandas_types) for v in value]
        if isinstance(value, tuple):
            return tuple(
                DictBundle._plain_value(v, pandas_types) for v in value
            )
        return value

    @staticmethod
    def _to_dict(self) -> Dict[str, Any]:
        """Nested dict view of a generated instance | 生成实例的嵌套字典视图"""
        pandas_types = type(self)._pandas_types
        return {
            k: DictBundle._plain_value(v, pandas_types)
            for k, v in self.__dict__.items()
        }

    @staticmethod
    def _repr(self) -> str:
        """Field structure of a generated instance | 生成实例的字段结构"""
        def describe(value: Any, pandas_types: Tuple[type, ...]) -> Any:
            """Describes one field value | 描述单个字段值"""
            if isinstance(value, pandas_types):
                # 自定义 pandas_types 不一定有 shape, 不能假定
                shape = getattr(value, 'shape', None)
                if shape is None:
                    return f"<{type(value).__name__}>"
                return f"<{type(value).__name__} shape={shape}>"
            if getattr(type(value), DictBundle.flag, False):
                return describe_fields(value)
            if isinstance(value, dict):
                return {
                    k: describe(v, pandas_types) for k, v in value.items()
                }
            if isinstance(value, (list, tuple)):
                return f"<{type(value).__name__} len={len(value)}>"
            return type(value).__name__

        def describe_fields(instance: Any) -> Dict[str, Any]:
            """Describes one level of fields | 描述一层字段"""
            pandas_types = type(instance)._pandas_types
            return {
                k: describe(v, pandas_types)
                for k, v in instance.__dict__.items()
            }

        rendered = pformat(describe_fields(self), width=60, sort_dicts=False)
        return f"{type(self).__name__}:\n{rendered}"

    @staticmethod
    def _dispatch(self, attr: str) -> Any:
        """
        =======================================================================
        Dispatches a pandas API name onto every pandas field of a generated
        instance.

        Attribute lookup on a field name never reaches this hook, so field
        names always win over dispatched names. Names starting with an
        underscore are rejected to keep copy, deepcopy, pickle and IPython
        probing intact.

        Parameters
        ----------
        self : Any
            The generated instance.
        attr : str
            The attribute name looked up on the instance.

        Returns
        -------
        Any
            A callable applying the API per field when the name is callable on
            at least one pandas field, otherwise the mapped object itself.
        -----------------------------------------------------------------------
        把 pandas 的 API 名分发到生成实例的每个 pandas 字段上.

        字段名的属性访问不会进入这个钩子, 所以字段名永远优先于分发的名字. 下划线
        开头的名字一律拒绝, 以保证 copy, deepcopy, pickle 与 IPython 的探测不被
        劫持.

        参数
        ----
        self : Any
            生成类的实例.
        attr : str
            在实例上查找的属性名.

        返回
        ----
        Any
            该名称在至少一个 pandas 字段上可调用时, 返回逐字段执行该调用的函数;
            否则直接返回映射后的新对象.
-------------------------------------------------------------------------------
        """
        if attr.startswith('_'):
            raise AttributeError(
                f"<{type(self).__name__}> object has no attribute {attr!r}"
            )
        targets = {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, type(self)._pandas_types)
        }
        if not targets:
            raise AttributeError(
                f"<{type(self).__name__}> object has no attribute {attr!r}"
            )
        unsupported = [k for k, v in targets.items() if not hasattr(v, attr)]
        if unsupported:
            raise AttributeError(
                f"<{type(self).__name__}> fields {unsupported} do not support "
                f"attribute {attr!r}"
            )

        references = {k: getattr(v, attr) for k, v in targets.items()}
        if not any(callable(reference) for reference in references.values()):
            return replace(self, **references)

        def mapped(*args: Any, **kwargs: Any) -> Any:
            """Applies the call to every pandas field | 对所有 pandas 字段执行该调用"""
            updates = {}
            for key, value in targets.items():
                reference = getattr(value, attr)
                if callable(reference):
                    updates[key] = reference(*args, **kwargs)
                else:
                    updates[key] = reference
            return replace(self, **updates)

        return mapped

dict_to_dataclass = DictBundle.build_legacy
"""Temporary alias for existing call sites | 现有调用点的临时别名"""
