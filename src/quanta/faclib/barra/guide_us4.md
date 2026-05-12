# Barra USA4 Factor Guide (guide_us4.py)

Summary
-------
    This guide covers the `guide_us4.py` implementation, which exposes the 
    core Barra USA4 style factors and risk modeling interfaces.

Publicly Exposed Factors
------------------------
    I   Style Factors:
        -- `size()`: Market capitalization factor (log).
        -- `bm()`: Book-to-market ratio (neutralized against size).
        -- `momentum()`: Exponentially weighted relative return momentum.
        -- `resid_volatility()`: Composite residual volatility factor (DASTD, CMRA, HSIGMA).
        -- `liquidity()`: Aggregated average turnover rates.
        -- `earnings()`: Composite profitability factor (CP, EP, EXEP).
        -- `growth()`: Earnings and revenue growth trends calculated via OLS.
        -- `leverage()`: Composite of Market, Debt-to-Asset, and Book leverage.

    II  Interfaces:
        -- `bench(code, weight)`: Benchmark factor retrieval.
        -- `neutral(df, factors_name)`: Neutralize custom alpha factors against standard Barra risk factors.

---

# Barra USA4 因子指南 (guide_us4.py)

概要
----
    本指南涵盖 `guide_us4.py` 的实现, 该模块暴露了用于风险建模和策略对比的
    核心 Barra USA4 风格因子.

公开暴露因子
------------
    I   风格因子:
        -- `size()`: 对数市值因子.
        -- `bm()`: 账面市值比 (针对市值中性化).
        -- `momentum()`: 指数加权相对收益动量.
        -- `resid_volatility()`: 复合残差波动率因子 (DASTD, CMRA, HSIGMA).
        -- `liquidity()`: 汇总的平均换手率因子.
        -- `earnings()`: 复合盈利因子 (CP, EP, EXEP).
        -- `growth()`: 通过 OLS 计算的盈利与营收增长趋势因子.
        -- `leverage()`: 市场、资产负债及账面杠杆的综合因子.

    II  接口:
        -- `bench(code, weight)`: 基准因子获取.
        -- `neutral(df, factors_name)`: 将自定义 Alpha 因子针对标准 Barra 风险因子进行中性化处理.
