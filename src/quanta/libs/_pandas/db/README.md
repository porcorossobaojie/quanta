# Database Connectivity (db)

Summary
-------
    The `db` submodule provides a convenient interface for persisting 
    Pandas DataFrames directly to the configured database (e.g., DuckDB, 
    MySQL). It integrates with the core `quanta.libs.db` engine to 
    provide high-performance data writing with support for various 
    existence strategies.

    Accessed via the `.db` accessor on a Pandas DataFrame.

Key Functions
-------------
    I   Data Persistence
        -- `.db.write()`: Writes the DataFrame to a database table. Supports 
            'append', 'replace', and 'fail' modes for existing tables. 
            Handles automatic schema mapping through the underlying engine.

Examples
--------
    # Write factor results to a DuckDB table
    factor_df.db.write(table_name='alpha_momentum', if_exists='append')

    # Overwrite an existing configuration table
    config_df.db.write(table_name='strategy_params', if_exists='replace')

---

# 数据库连接 (db)

概要
----
    `db` 子模块提供了一个便捷的接口, 用于将 Pandas DataFrame 直接持久化
    至配置的数据库 (如 DuckDB, MySQL). 它与核心的 `quanta.libs.db` 引擎
    集成, 提供支持多种存在策略的高性能数据写入.

    通过 Pandas DataFrame 上的 `.db` 访问器进行调用.

核心函数
--------
    I   数据持久化
        -- `.db.write()`: 将 DataFrame 写入数据库表. 对于已存在的表, 
            支持 'append' (追加), 'replace' (替换) 和 'fail' (失败) 
            模式. 通过底层引擎处理自动模式映射.

示例
----
    # 将因子结果写入 DuckDB 表
    factor_df.db.write(table_name='alpha_momentum', if_exists='append')

    # 覆盖现有的配置表
    config_df.db.write(table_name='strategy_params', if_exists='replace')
