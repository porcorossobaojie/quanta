# Trading & Execution (trade)

Summary
-------
    The `trade` module handles the execution and post-trade processing 
    infrastructure of the `quanta` framework. It provides tools for 
    managing trading accounts, processing settlement files, and 
    integrating with various brokers through a unified pipeline.

    The module ensures that theoretical portfolio changes are correctly 
    translated into orders and that actual trade executions are reconciled 
    via settlement processes.

Key Components
--------------
    I   Account Management (`account/main.py`)
        -- `main`: Manages strategy-specific trading accounts. It handles 
            the local directory structure for orders and settlement files 
            and provides methods for reading position data.

    II  Trading Pipelines (`pipline/`)
        -- Provides broker-specific implementations (e.g., `tonghua/`) for 
            reading and writing trade data. 
        -- Ensures that data formats from different brokers are standardized 
            before being consumed by the account manager.

Workflow
--------
    1.  **Environment Initialization**: The account manager automatically 
        sets up the required folder structure based on the strategy name.
    2.  **Order Generation**: Strategies generate order files which are 
        stored in the strategy's `order/` directory.
    3.  **Settlement Reconciliation**: The `settle()` method reads 
        settlement files provided by the broker to update the current 
        portfolio state, ensuring standardized asset coding and correct 
        position accounting.

---

# 交易与执行 (trade)

概要
----
    `trade` 模块处理 `quanta` 框架的执行和盘后处理基础设施. 它提供了
    管理交易账户, 处理结算文件以及通过统一流水线与各种经纪商集成的工具.

    该模块确保将理论上的投资组合变更正确转换为订单, 并且通过结算过程
    核对实际的交易执行情况.

核心组件
--------
    I   账户管理 (`account/main.py`)
        -- `main`: 管理特定策略的交易账户. 它处理订单和结算文件的本地
            目录结构, 并提供读取持仓数据的方法.

    II  交易流水线 (`pipline/`)
        -- 提供针对特定经纪商 (如 `tonghua/`) 的实现, 用于读取和写入
            交易数据. 
        -- 确保在被账户管理器调用之前, 来自不同经纪商的数据格式已得到
            标准化处理.

工作流程
--------
    1.  **环境初始化**: 账户管理器根据策略名称自动设置所需的文件夹结构.
    2.  **订单生成**: 策略生成订单文件, 并存储在策略的 `order/` 目录中.
    3.  **结算核对**: `settle()` 方法读取经纪商提供的结算文件以更新
        当前的投资组合状态, 确保标准的资产编码和正确的持仓核算.
