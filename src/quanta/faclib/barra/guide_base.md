# Barra Base Factor Guide (_base.py)

Summary
-------
    The `faclib.barra._base` module provides the foundational engine and low-level 
    methods for calculating Barra risk model components.

Core Base Factors & Utilities
-----------------------------
    I   Base Calculations:
        -- `size()`: Log market capitalization.
        -- `bm()`: Book-to-market ratio (size-neutralized).
        -- `non_size()`: Non-linear size factor.
        -- `beta(...)`: Market beta calculation (vectorized rolling regression).
        -- `hsigma(...)`: Historical residual volatility calculation.
        -- `dastd(...)`: Daily standard deviation (returns).
        -- `cmra(...)`: Cumulative range of adjusted returns.

    II  Turnover & Growth Factors:
        -- `month_turnover()` / `quarter_turnover()` / `annual_turnover()`: Liquidity metrics.
        -- `annul_weight_turnover()`: Annual weighted turnover.
        -- `short_term_reversal()`: Short-term price reversal factor.
        -- `seasonal(...)`: Seasonal trend factor.
        -- `industry_momentum(...)`: Momentum aggregated at the industry level.

    III Specialized Financial Factors:
        -- `market_leverage()` / `book_leverage()` / `debt_to_asset_ratio()`: Leverage metrics.
        -- `variation_in_sales()` / `variation_to_earnings()` / `variation_to_cashflow()`: Metric volatility.
        -- `accr_balancesheet()` / `accr_cashflow()`: Accruals calculations.
        -- `asset_turnover()` / `gross_profit()` / `roa()`: Performance metrics.
        -- `asset_growth()` / `invest_growth()`: Growth metrics.

---

# Barra 基础因子指南 (_base.py)

概要
----
    `faclib.barra._base` 模块提供了计算 Barra 风险模型组件的基础引擎和底层方法.

核心基础因子与工具
------------------
    I   基础计算:
        -- `size()`: 对数市值.
        -- `bm()`: 账面市值比 (市值中性化).
        -- `non_size()`: 非线性市值因子.
        -- `beta(...)`: 市场 Beta (向量化滚动回归).
        -- `hsigma(...)`: 历史残差波动率.
        -- `dastd(...)`: 日收益率标准差.
        -- `cmra(...)`: 累积相对收益范围.

    II  换手率与增长因子:
        -- `month_turnover()` / `quarter_turnover()` / `annual_turnover()`: 流动性指标.
        -- `annul_weight_turnover()`: 年度加权换手率.
        -- `short_term_reversal()`: 短期价格反转因子.
        -- `seasonal(...)`: 季节性趋势因子.
        -- `industry_momentum(...)`: 行业层面动量.

    III 专业财务因子:
        -- `market_leverage()` / `book_leverage()` / `debt_to_asset_ratio()`: 杠杆指标.
        -- `variation_in_sales()` / `variation_to_earnings()` / `variation_to_cashflow()`: 指标波动度.
        -- `accr_balancesheet()` / `accr_cashflow()`: 应计项目计算.
        -- `asset_turnover()` / `gross_profit()` / `roa()`: 绩效指标.
        -- `asset_growth()` / `invest_growth()`: 增长指标.
