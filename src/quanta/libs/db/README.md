# Database Engine (db)

Summary
-------
    The `db` module is the core persistence layer of the `quanta` framework. 
    It provides a unified interface for interacting with different database 
    engines, primarily DuckDB (for local, high-performance analytical storage) 
    and MySQL (for centralized, multi-user access).

    The module utilizes a facade pattern, where the `main` class dynamically 
    inherits from the configured engine implementation.

Key Components
--------------
    I   Engine Facade (`main.py`)
        -- Automatically selects the appropriate engine (DuckDB or MySQL) 
            based on global settings.
        -- Provides a consistent API for reading, writing, and managing tables.

    II  Engine Implementations (`_engines/`)
        -- `DuckDB.py`: Specialized implementation for DuckDB, optimized for 
            fast analytical queries and local `.duckdb` files.
        -- `MySQL.py`: Implementation for MySQL/MariaDB, supporting indexing, 
            partitioning, and remote connections via SQLAlchemy/pymysql.
        -- `meta.py`: Base class defining the mandatory interface and 
            utility methods (like parameter merging and timing decorators).

    III Data Type Standardization (`_data_type_standard/`)
        -- Ensures consistent mapping between Python/Pandas types and 
            database-specific SQL types across different engines.

Core API
--------
    -- `read()`: Executes SQL queries and returns a Pandas DataFrame. Supports 
        automatic parameterization and performance timing.
    -- `write()`: Persists a DataFrame to a table. Handles table creation, 
        existence checks, and data type mapping.
    -- `command()`: Executes raw SQL commands.
    -- `table_exist()`: Checks for the existence of a specific table in the 
        schema.
    -- `create_table()` / `drop_table()`: DDL operations for table management.

---

# 数据库引擎 (db)

概要
----
    `db` 模块是 `quanta` 框架的核心持久层. 它为与不同数据库引擎交互提供
    了统一接口, 主要包括 DuckDB (用于本地, 高性能分析存储) 和 MySQL 
    (用于中心化, 多用户访问).

    该模块采用外观模式, `main` 类根据配置设置动态继承相应的引擎实现.

核心组件
--------
    I   引擎外观 (`main.py`)
        -- 根据全局设置自动选择合适的引擎 (DuckDB 或 MySQL).
        -- 提供一致的 API 用于读取, 写入和管理表.

    II  引擎实现 (`_engines/`)
        -- `DuckDB.py`: 针对 DuckDB 的专用实现, 针对快速分析查询和本地 
            `.duckdb` 文件进行了优化.
        -- `MySQL.py`: 针对 MySQL/MariaDB 的实现, 支持通过 SQLAlchemy/
            pymysql 进行索引, 分区和远程连接.
        -- `meta.py`: 基类, 定义了强制性接口和实用方法 (如参数合并和计时
            装饰器).

    III 数据类型标准化 (`_data_type_standard/`)
        -- 确保在不同引擎之间, Python/Pandas 类型与特定数据库的 SQL 类型
            具有一致的映射.

核心 API
--------
    -- `read()`: 执行 SQL 查询并返回 Pandas DataFrame. 支持自动参数化
        和性能计时.
    -- `write()`: 将 DataFrame 持久化到表中. 处理表创建, 存在性检查和
        数据类型映射.
    -- `command()`: 执行原始 SQL 命令.
    -- `table_exist()`: 检查模式中是否存在指定表.
    -- `create_table()` / `drop_table()`: 用于表管理的 DDL 操作.
