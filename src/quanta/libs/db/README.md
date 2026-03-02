# Database Abstraction Layer

Summary
-------
    The `db` module provides a high-level, engine-agnostic abstraction layer 
    for database operations. It ensures that the rest of the `quanta` 
    framework can interact with different storage backends (e.g., DuckDB, 
    MySQL) using a unified set of internal protocols.

    The architecture is built on dynamic dispatch and attribute-driven 
    configuration.

Module Structure
----------------
    The database layer is organized into three main components:

    I   _engines/
        Contains specific driver implementations.
        -- meta.py: The foundation class providing parameter merging (MRO 
           analysis), timing decorators, and instance state management.
        -- DuckDB.py: Optimized for local analytical workloads.
        -- MySQL.py: Optimized for production-grade persistent storage.

    II  _data_type_standard/
        Manages the translation matrix between Pandas/NumPy types and 
        database-specific SQL types, ensuring cross-engine integrity.

    III main.py (Facade)
        The primary entry point that automatically selects the active engine 
        based on `libs.yaml` settings.

Internal Protocols
------------------
    All engines implement a standardized set of internal methods:

    -- `__read__`: Generic data retrieval with built-in timing.
    -- `__write__`: Optimized batch writing for DataFrames.
    -- `__create_table__`: Schema-aware table initialization with partitioning 
       support (MySQL).
    -- `__table_exist__`: Existence check across different schemas.

Maintenance
-----------
    To add a new engine:
        1. Inherit from `_engines.meta.main`.
        2. Implement the standard internal protocol methods.
        3. Register the engine mapping in `libs.yaml` and `db/main.py`.

---

# 数据库抽象层 (中文版)

概要
----
    `db` 模块为数据库操作提供了一个高级的、与引擎无关的抽象层. 它确保 
    `quanta` 框架的其他部分可以使用一套统一的内部协议与不同的存储后端 
    (如 DuckDB, MySQL) 进行交互.

    该架构基于动态分派和属性驱动配置构建.

模块结构
--------
    数据库层由三个主要部分组成:

    I   _engines/
        包含具体的驱动实现.
        -- meta.py: 基础类, 提供参数合并 (MRO 分析)、计时装饰器及
           实例状态管理.
        -- DuckDB.py: 针对本地分析型任务进行了优化.
        -- MySQL.py: 针对生产级持久化存储进行了优化.

    II  _data_type_standard/
        管理 Pandas/NumPy 类型与数据库特定 SQL 类型之间的转换矩阵, 
        确保跨引擎的数据完整性.

    III main.py (门面)
        主要入口点, 根据 `libs.yaml` 设置自动选择活动的引擎.

内部协议
--------
    所有引擎均实现了一套标准化的内部方法:

    -- `__read__`: 具有内置计时功能的通用数据检索.
    -- `__write__`: 针对 DataFrame 优化的批量写入.
    -- `__create_table__`: 感知模式的表初始化, 支持分区 (MySQL).
    -- `__table_exist__`: 跨不同模式的表存在性检查.

维护指南
--------
    添加新引擎的步骤:
        1. 继承自 `_engines.meta.main`.
        2. 实现标准的内部协议方法.
        3. 在 `libs.yaml` 和 `db/main.py` 中注册引擎映射.
