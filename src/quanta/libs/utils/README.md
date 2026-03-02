# Utility Collections

Summary
-------
    The `utils` module provides a collection of low-level, reusable helper 
    functions and decorators that support the infrastructure and meta-logic 
    of the `quanta` framework. 

    These utilities are designed to be generic and dependency-light, 
    focusing on dictionary manipulation, class introspection, and 
    documentation management.

Module Structure
----------------
    The utilities are categorized into base helpers and decorators:

    I   _base.py (Core Helpers)
        Functions for data structure and class attribute management.
        -- `filter_parents_class_attrs`: Extracts merged attributes from a 
           class and its MRO, supporting complex inheritance hierarchies.
        -- `merge_dicts`: Deep-merges multiple dictionaries with priority 
           overriding and None-value skipping.
        -- `flatten_list`: Recursively flattens nested list structures.

    II  _decorator.py (Decorators)
        Advanced decorators for performance and maintainability.
        -- `doc_inherit`: Copies docstrings from source functions to target 
           methods to ensure documentation synchronization.
        -- `timing_decorator`: Precision execution timer for database and 
           data source operations.

Key Implementation
------------------
    Inheritance Introspection:
        The framework heavily uses `utils` to implement its "Attribute-Driven" 
        design, allowing child classes to automatically inherit and merge 
        configurations from their parents.

---

# 工具函数集 (中文版)

概要
----
    `utils` 模块提供了一系列底层的、可重用的辅助函数和装饰器, 
    支撑着 `quanta` 框架的基础设施和元逻辑.

    这些工具旨在保持通用性和轻量级依赖, 侧重于字典操作、类内省以及
    文档管理.

模块结构
--------
    工具集分为基础辅助函数和装饰器:

    I   _base.py (核心辅助函数)
        用于数据结构和类属性管理的函数.
        -- `filter_parents_class_attrs`: 从类及其 MRO 中提取合并属性, 
           支持复杂的继承层次结构.
        -- `merge_dicts`: 深度合并多个字典, 支持优先级覆盖和 None 值跳过.
        -- `flatten_list`: 递归展平嵌套的列表结构.

    II  _decorator.py (装饰器)
        用于性能监控和维护的高级装饰器.
        -- `doc_inherit`: 从源函数复制文档字符串至目标方法, 
           确保文档同步.
        -- `timing_decorator`: 为数据库和数据源操作设计的精密执行计时器.

核心实现
--------
    继承内省:
        框架大量使用 `utils` 来实现其 "属性驱动" 设计, 允许子类自动继承
        并合并来自父类的配置信息.
