# Data Management

Summary
-------
    This directory serves as the core data engine for the `quanta` framework. 
    It manages the entire ETL (Extract, Transform, Load) lifecycle, 
    synchronizing professional financial data from remote providers to local 
    analytical databases (DuckDB/MySQL).

    Currently, the module is deeply integrated with JoinQuant (JQData).

Module Structure
----------------
    The `data` module is organized by provider and update logic:

    I  joinquant/
        The primary data source provider.
        -- meta: Base classes and common standardization logic.
        -- id_table: Incremental update logic based on unique IDs (e.g., 
           Financial Statements, Corporate Actions).
        -- dt_table: Time-series update logic based on trade dates (e.g., 
           EOD Prices, Valuation Indicators).

Core Interface
--------------
    The simplest way to maintain your local database is through the 
    top-level `daily` function:

        import quanta.data as data
        
        # Synchronize all configured tables
        data.daily()

    This function automatically:
        1. Authenticates with JQData using your local credentials.
        2. Scans local tables to find the last updated record.
        3. Fetches missing data, standardizes it, and appends it locally.

---

# 数据管理 (中文版)

概要
----
    本目录是 `quanta` 框架的核心数据引擎. 它管理完整的 ETL (抽取, 转换, 
    加载) 生命周期, 将专业金融数据从远程供应商同步至本地分析型数据库 
    (DuckDB/MySQL).

    目前, 该模块深度集成了聚宽 (JoinQuant/JQData) 数据源.

模块结构
--------
    `data` 模块按供应商和更新逻辑进行组织:

    I  joinquant/
        主要数据供应商.
        -- meta: 基础类及通用标准化逻辑.
        -- id_table: 基于唯一 ID 的增量更新逻辑 (如: 财务报表, 
           分红送股).
        -- dt_table: 基于交易日期的序列更新逻辑 (如: 日末行情, 
           估值指标).

核心接口
--------
    维护本地数据库最简单的方法是使用顶层的 `daily` 函数:

        import quanta.data as data
        
        # 同步所有配置的表格
        data.daily()

    该函数会自动执行以下操作:
        1. 使用本地凭据通过 JQData 身份验证.
        2. 扫描本地表以找到最后一条更新记录.
        3. 获取缺失数据, 进行标准化处理并追加至本地.
