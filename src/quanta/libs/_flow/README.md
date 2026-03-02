# Research Flow Layer

Summary
-------
    The `flow` module is the high-level research engine of the `quanta` 
    framework. It abstracts complex database queries into intuitive, 
    attribute-driven interfaces, enabling researchers to focus on factor 
    logic and strategy construction rather than data fetching.

    It consists of unified data access objects and powerful Pandas extensions.

Module Structure
----------------
    The module is divided into core access and functional extensions:

    I  _main/
        Defines the primary research entry points (e.g., astock, afund).
        -- Data Access: Unified `__call__` interface for cross-table columns.
        -- Finance: Specialized handling for YTD-to-Quarterly conversion.
        -- Metadata: Built-in fuzzy search for columns via `.help()`.

    II _extra_pandas/
        Registers the `f` accessor to pandas.DataFrame/Series.
        -- Filters: `listing`, `not_st`, `tradestatus`.
        -- Analysis: `ic`, `ir`, `trend`.
        -- Backtest: `port`, `test` (Quick group Construct & Vectorized BT).

Core Interface Examples
-----------------------
    I  Unified Data Fetching:
        
        from quanta import flow
        
        # Fetch price and financial columns simultaneously
        df = flow.astock("close_adj")
        
        # Fuzzy search for revenue-related fields
        flow.astock.help("revenue")

    II Chained Factor Analysis:

        # Filter, calculate IC, and run quick test in one flow
        (df.f.filtered()
           .f.port())

---

# 研究流层 (中文版)

概要
----
    `flow` 模块是 `quanta` 框架的高级研究引擎. 它将复杂的数据库查询抽象为
    直观的、属性驱动的接口, 使研究员能够专注于因子逻辑和策略构建, 而非
    数据获取细节.

    该模块由统一的数据访问对象和强大的 Pandas 扩展组成.

模块结构
--------
    该模块分为核心访问层和功能扩展层:

    I  _main/
        定义主要的研究入口 (如: astock, afund).
        -- 数据访问: 跨表字段的统一 `__call__` 接口.
        -- 财务处理: 专用的财务报表累计值转单季度值逻辑.
        -- 元数据: 通过 `.help()` 内置字段模糊搜索功能.

    II _extra_pandas/
        将 `f` 访问器注册至 pandas.DataFrame/Series.
        -- 过滤工具: `listing` (上市), `not_st` (剔除ST), `tradestatus` (交易状态).
        -- 因子分析: `ic`, `ir`, `trend` (趋势).
        -- 策略回测: `port` (分组收益), `test` (向量化快速回测).

核心接口示例
------------
    I  统一数据获取:
        
        from quanta import flow
        
        # 同时获取复权价格和财务字段
        df = flow.astock("close_adj")
        
        # 模糊搜索营业收入相关的字段
        flow.astock.help("revenue")

    II 链式因子分析:

        # 在一个流中完成过滤、计算 IC 和快速回测
        (df.f.filtered()
           .f.port())
