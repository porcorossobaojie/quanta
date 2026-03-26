# Factor Library (faclib)

Summary
-------
    The `faclib` module contains the definitions and implementations of 
    quantitative factors used in the `quanta` framework. It includes 
    standard risk models (like Barra), alpha factors, and custom 
    technical/fundamental indicators.

    The library is structured with a base meta-class providing common utilities 
    and specific submodules for different factor families.

Key Components
--------------
    I   Base Factor Class (`_base/main.py`)
        -- `meta`: Provides core utilities such as `bench()` for retrieving 
            benchmark returns and handling configuration mappings.

    II  Barra Risk Factors (`barra/`)
        -- Implements standard Barra-style factors including Size, 
            Non-linear Size, and Beta.
        -- Utilizes high-performance vectorized operations (NumPy, Numexpr) 
            for rolling regressions and cross-sectional analysis.

Factor Design Principles
------------------------
    1.  **Vectorization**: All factor calculations are designed to be 
        vectorized over time and assets using Pandas and NumPy.
    2.  **Neutralization**: Core risk factors are often neutralized against 
        each other (e.g., Book-to-Market neutralized against Size).
    3.  **Efficiency**: Heavy calculations (like rolling beta) use optimized 
        approaches like `einsum` or `array_roll` to minimize execution time.

---

# 因子库 (faclib)

概要
----
    `faclib` 模块包含 `quanta` 框架中使用的量化因子的定义和实现. 它包括
    标准风险模型 (如 Barra), Alpha 因子以及自定义的技术/基本面指标.

    该库的结构包含一个提供通用工具的基类 (meta-class), 以及针对不同因子
    家族的具体子模块.

核心组件
--------
    I   因子基类 (`_base/main.py`)
        -- `meta`: 提供核心实用工具, 如用于检索基准收益和处理配置映射
            的 `bench()` 函数.

    II  Barra 风险因子 (`barra/`)
        -- 实现标准的 Barra 风格因子, 包括市值 (Size), 非线性市值 
            (Non-linear Size) 和贝塔 (Beta).
        -- 利用高性能向量化操作 (NumPy, Numexpr) 进行滚动回归和横截面分析.

因子设计原则
------------
    1.  **向量化**: 所有因子计算均设计为利用 Pandas 和 NumPy 在时间和资产
        维度上进行向量化.
    2.  **中性化**: 核心风险因子通常相互进行中性化处理 (例如, 账面市值比
        针对市值进行中性化).
    3.  **效率**: 沉重的计算 (如滚动贝塔) 使用 `einsum` 或 `array_roll` 
        等优化方法, 以最大限度地减少执行时间.
