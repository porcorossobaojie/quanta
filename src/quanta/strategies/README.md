# Strategies (strategies)

Summary
-------
    The `strategies` module serves as the primary workspace for implementing 
    and backtesting quantitative trading strategies. It provides the 
    foundational structures and interfaces needed to define investment logic 
    that interacts with the broader `quanta` framework.

Key Components
--------------
    I   Strategy Meta-Framework (`meta/`)
        -- `main.py`: Contains the base strategy class `main`, defining the
            standard factor-driven workflow: pool construction, factor
            ranking, sell/hold/buy decisions, rebalancing, and order writing.
        -- Provides a standardized approach to manage strategy parameters and
            lifecycle, ensuring consistency across different implementations.

Implementation
--------------
    Strategies inherit from the meta-framework to ensure they are compatible
    with the `quanta` data pipelines and trade management tools.

        from quanta.strategies.meta import main as StrategyBase

        class MyMomentumStrategy(StrategyBase):
            @classmethod
            def factor(cls):
                # Override with a custom factor, e.g., Barra momentum
                return faclib.barra.us4.momentum()

    The base workflow is then driven by the inherited hooks (`pool`,
    `ranker`, `settle`, `enbuy`, `rebalance`, `signal`, `write`).

---

# 策略 (strategies)

概要
----
    `strategies` 模块是实现和回测量化交易策略的主要工作区. 它提供了定义
    投资逻辑所需的基础结构和接口, 这些逻辑与更广泛的 `quanta` 框架进行
    交互.

核心组件
--------
    I   策略元框架 (`meta/`)
        -- `main.py`: 包含基础策略类 `main`, 定义了标准的因子驱动流程: 
            股票池构建, 因子排序, 卖出/持有/买入决策, 再平衡及下单写入.
        -- 提供了一种标准化的方法来管理策略参数和生命周期, 确保不同实现
            之间的一致性.

实现与调用
----------
    策略继承自元框架, 以确保它们与 `quanta` 数据流水线和交易管理工具
    兼容.

        from quanta.strategies.meta import main as StrategyBase

        class MyMomentumStrategy(StrategyBase):
            @classmethod
            def factor(cls):
                # 覆写为自定义因子, 例如 Barra 动量因子
                return faclib.barra.us4.momentum()

    基础工作流由继承的钩子方法驱动 (`pool`, `ranker`, `settle`, `enbuy`,
    `rebalance`, `signal`, `write`).
