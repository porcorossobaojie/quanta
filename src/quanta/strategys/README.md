# Strategies (strategys)

Summary
-------
    The `strategys` module serves as the primary workspace for implementing 
    and backtesting quantitative trading strategies. It provides the 
    foundational structures and interfaces needed to define investment logic 
    that interacts with the broader `quanta` framework.

Key Components
--------------
    I   Strategy Meta-Framework (`meta/`)
        -- `main.py`: Contains the base strategy class or template, defining 
            standard hooks for strategy initialization, signal generation, 
            and trade execution logging.
        -- Provides a standardized approach to manage strategy parameters and 
            lifecycle, ensuring consistency across different implementations.

Implementation
--------------
    Strategies inherit from the meta-framework to ensure they are compatible 
    with the `quanta` data pipelines and trade management tools. 

        from quanta.strategys.meta import StrategyBase

        class MyMomentumStrategy(StrategyBase):
            def on_signal(self, context):
                # Implement custom signal logic
                pass

---

# 策略 (strategys)

概要
----
    `strategys` 模块是实现和回测量化交易策略的主要工作区. 它提供了定义
    投资逻辑所需的基础结构和接口, 这些逻辑与更广泛的 `quanta` 框架进行
    交互.

核心组件
--------
    I   策略元框架 (`meta/`)
        -- `main.py`: 包含基础策略类或模板, 定义了策略初始化、信号生成
            和交易执行记录的标准钩子.
        -- 提供了一种标准化的方法来管理策略参数和生命周期, 确保不同实现
            之间的一致性.

实现与调用
----------
    策略继承自元框架, 以确保它们与 `quanta` 数据流水线和交易管理工具
    兼容.

        from quanta.strategys.meta import StrategyBase

        class MyMomentumStrategy(StrategyBase):
            def on_signal(self, context):
                # 实现自定义信号逻辑
                pass
