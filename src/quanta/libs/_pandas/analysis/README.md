# Quantitative Analysis (analysis)

Summary
-------
    The `analysis` submodule provides tools for evaluating strategy performance 
    and risk metrics. It includes functions for drawdown analysis, risk-adjusted 
    returns (Sharpe ratio), and factor quality assessment (linearity/effectiveness).

    Accessed via the `.analysis` accessor on Pandas DataFrame or Series.

Key Functions
-------------
    I   Risk & Return Metrics
        -- `.analysis.maxdown()`: Calculates the maximum drawdown period, 
            start/end values, and percentage. Supports both cumulative and 
            periodic returns.
        -- `.analysis.sharpe()`: Computes the annualized Sharpe ratio, 
            providing a measure of risk-adjusted return.

    II  Factor Evaluation
        -- `.analysis.effective()`: Quantifies factor linearity by analyzing 
            returns across ranked groups. Higher values indicate more 
            reliable monotonic relationships between factor values and returns.

Examples
--------
    # Calculate Sharpe ratio for strategy returns
    sharpe_ratio = returns_df.analysis.sharpe()

    # Get maximum drawdown statistics
    mdd_stats = nav_df.analysis.maxdown()

    # Evaluate factor linearity from group returns
    linearity_score = group_rets.analysis.effective()

---

# 量化分析 (analysis)

概要
----
    `analysis` 子模块提供了用于评估策略绩效和风险指标的工具. 它包括用于
    回撤分析, 风险调整收益 (夏普比率) 和因子质量评估 (线性/有效性) 的
    函数.

    通过 Pandas DataFrame 或 Series 上的 `.analysis` 访问器进行调用.

核心函数
--------
    I   风险与收益指标
        -- `.analysis.maxdown()`: 计算最大回撤周期, 起始/结束数值以及
            百分比. 支持累计收益和周期性收益.
        -- `.analysis.sharpe()`: 计算年化夏普比率, 提供风险调整后的
            收益衡量指标.

    II  因子评估
        -- `.analysis.effective()`: 通过分析排名分组的收益来量化因子
            线性性. 较高的数值表示因子值与收益之间具有更可靠的单调关系.

示例
----
    # 计算策略收益的夏普比率
    sharpe_ratio = returns_df.analysis.sharpe()

    # 获取最大回撤统计数据
    mdd_stats = nav_df.analysis.maxdown()

    # 从分组收益评估因子线性性
    linearity_score = group_rets.analysis.effective()
