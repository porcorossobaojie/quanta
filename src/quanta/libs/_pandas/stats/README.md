# Statistical Modeling (stats)

Summary
-------
    The `stats` submodule provides advanced statistical tools optimized for 
    quantitative finance. It includes high-performance implementations of 
    OLS/WLS regression, cross-sectional standardization, and factor 
    neutralization (removing exposure to specific factors like industry or size).

    Accessed via the `.stats` accessor on Pandas DataFrame or Series.

Key Functions
-------------
    I   Data Normalization
        -- `.stats.standard()`: Standardizes data using Gaussian (Z-score) 
            or uniform mapping. Supports cross-sectional or time-series axes.
        -- `.stats.const()`: Generates dummy variables (indicator matrices) 
            for categorical data.

    II  Regression & Neutralization
        -- `.stats.OLS()`: Ordinary or Weighted Least Squares regression 
            with support for rolling windows.
        -- `.stats.neutral()`: Multi-variate neutralization. Returns a 
            specialized result object containing residuals, t-statistics, 
            and R-squared values. Optimized for high-dimensional factor data.

Examples
--------
    # Standardize a factor cross-sectionally
    std_factor = factor_df.stats.standard(method='gauss')

    # Neutralize a factor against industry dummies and market cap
    neutral_obj = factor_df.stats.neutral(industry=ind_df, size=size_df)
    resid_factor = neutral_obj.resid

---

# 统计建模 (stats)

概要
----
    `stats` 子模块提供了针对量化金融优化的先进统计工具. 它包括 OLS/WLS 
    回归, 横截面标准化和因子中性化 (消除行业或市值等特定因子的暴露) 的
    高性能实现.

    通过 Pandas DataFrame 或 Series 上的 `.stats` 访问器进行调用.

核心函数
--------
    I   数据规范化
        -- `.stats.standard()`: 使用高斯 (Z-score) 或均匀映射进行数据
            标准化. 支持横截面或时间序列轴.
        -- `.stats.const()`: 为分类数据生成虚拟变量 (指示矩阵).

    II  回归与中性化
        -- `.stats.OLS()`: 普通或加权最小二乘回归, 支持滚动窗口.
        -- `.stats.neutral()`: 多元中性化. 返回一个专门的结果对象, 
            包含残差, t 统计量和 R 方值. 针对高维因子数据进行了优化.

示例
----
    # 对因子进行横截面标准化
    std_factor = factor_df.stats.standard(method='gauss')

    # 针对行业虚拟变量和市值对因子进行中性化处理
    neutral_obj = factor_df.stats.neutral(industry=ind_df, size=size_df)
    resid_factor = neutral_obj.resid
