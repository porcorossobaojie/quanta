# Configuration Definitions

Summary
-------
    This directory contains the central configuration layer for the `quanta`
    framework. These YAML files define the "attribute-driven" logic of the
    system, mapping remote data structures to local standards.

    All configurations are managed and loaded via `quanta.config.settings`.

data.yaml
---------
    Defines the bridge between JoinQuant (JQData) and the local database:
        -- public_keys: Global alignment settings (e.g., trade_dt, time_bias).
        -- transform: Field renaming mapping for standardization.
        -- id_table: Logic for incremental ID-based fetching (Financials).
        -- dt_table: Logic for trade-date-based fetching (Prices/EOD).

flow.yaml
---------
    Configures high-level research abstractions and filtering logic:
        -- listing: Criteria for listing/delisting date detection.
        -- status: Field mappings for ST and tradability status.
        -- trade_keys: Standard keys for backtesting (Close, VWAP, Returns).
        -- index_mapping: Quick lookup for index constituents.

libs.yaml
---------
    Defines infrastructure and engine-level settings:
        -- DuckDB: Local analytical storage settings (Path, Schema).
        -- MySQL: Production database connection and credentials.
        -- data_type: Type mapping matrix for cross-engine compatibility.

Implementation
--------------
    Configurations are accessed as attributes for high efficiency:

        from quanta.config import settings
        
        # Access data settings
        data_cfg = settings('data')
        trade_dt = data_cfg.public_keys.recommand_settings.key.trade_dt

---

# 配置定义 (中文版)

概要
----
    本目录包含 `quanta` 框架的核心配置层. 这些 YAML 文件定义了系统的
    "属性驱动" 逻辑, 将远程数据结构映射为本地标准.

    所有配置均通过 `quanta.config.settings` 统一管理和加载.

data.yaml
---------
    定义聚宽 (JQData) 与本地数据库之间的桥梁:
        -- public_keys: 全局对齐设置 (如 trade_dt, time_bias).
        -- transform: 用于标准化的字段重命名映射.
        -- id_table: 基于增量 ID 的获取逻辑 (财务报表).
        -- dt_table: 基于交易日期的获取逻辑 (价格/日末数据).

flow.yaml
---------
    配置高层研究抽象及过滤逻辑:
        -- listing: 上市/退市日期检测标准.
        -- status: ST 状态及可交易性状态字段映射.
        -- trade_keys: 回测标准键 (收盘价, VWAP, 收益率).
        -- index_mapping: 指数成份股快速查询字典.

libs.yaml
---------
    定义基础设施及引擎级设置:
        -- DuckDB: 本地分析型存储设置 (路径, Schema).
        -- MySQL: 生产数据库连接及凭据.
        -- data_type: 跨引擎兼容性的类型映射矩阵.

实现与加载
----------
    配置以属性方式访问, 确保高效引用:

        from quanta.config import settings
        
        # 访问数据配置
        data_cfg = settings('data')
        trade_dt = data_cfg.public_keys.recommand_settings.key.trade_dt
