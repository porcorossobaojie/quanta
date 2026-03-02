# Analytical Extensions (Pandas)

Summary
-------
    The `_pandas` module is the computational core of the `quanta` framework. 
    It extends standard `pandas.DataFrame` and `pd.Series` with high-performance, 
    vectorized functions tailored for quantitative finance.

    By leveraging Pandas accessors, these tools integrate seamlessly into 
    your existing data workflows.

Module Structure
----------------
    The module is organized into specialized accessors:

    I   gen/ (Generation)
        Tools for portfolio construction and signal processing.
        -- `.gen.group()`: Efficient multi-factor binning and ranking.
        -- `.gen.portfolio()`: Grouped return calculation with shift/lag support.

    II  rollings/ (Time-Series)
        High-performance rolling window operations.
        -- `.rollings(w).max(n).mean()`: Retrieves mean of the top N values in a rolling window.
        -- `.rollings(w).ts_rank()`: Calculates time-series percentage rank.

    III stats/ (Statistics)
        Statistical modeling and factor preparation.
        -- `.stats.neutral()`: Multi-variate linear regression for neutralization.
        -- `.stats.standard()`: Gaussian or uniform cross-sectional standardization.

    IV  analysis/ (Performance)
        Strategy evaluation and risk metrics.
        -- `.analysis.maxdown()`: Calculates detailed maximum drawdown statistics.
        -- `.analysis.sharpe()`: Annualized Sharpe ratio calculation.

    V   db/ & tools/ (Utilities)
        Infrastructure and data cleaning.
        -- `.db.write()`: Direct DataFrame persistence to DuckDB/MySQL.
        -- `.tools.log()`: Sign-preserved logarithmic transformation.

Core Interface Examples
-----------------------
    I   Factor Neutralization & Standardization:
        
        # Standardize and neutralize factor against industry and size
        factor = df.stats.standard().stats.neutral(industry=ind_df, size=size_df).resid

    II  Rolling Window Analysis:

        # Get the mean of the top 3 volume days over a 21-day rolling window
        top_vol = df.rollings(21).max(3).mean()

---

# 专业分析扩展 (中文版)

概要
----
    `_pandas` 模块是 `quanta` 框架的计算核心. 它通过高性能的向量化函数
    扩展了标准 `pandas.DataFrame` 和 `pd.Series`, 专门针对量化金融场景
    进行了优化.

    通过利用 Pandas Accessor 机制, 这些工具可以无缝集成到您现有的数据
    处理流中.

模块结构
--------
    该模块按专用的访问器进行组织:

    I   gen/ (生成工具)
        用于投资组合构建和信号处理的工具.
        -- `.gen.group()`: 高效的多因子分箱与排名.
        -- `.gen.portfolio()`: 支持位移和滞后的分组收益计算.

    II  rollings/ (时序运算)
        高性能滚动窗口操作.
        -- `.rollings(w).max(n).mean()`: 获取滚动窗口内的前 N 个最大值的均值.
        -- `.rollings(w).ts_rank()`: 计算时间序列百分比排名.

    III stats/ (统计建模)
        统计建模与因子预处理.
        -- `.stats.neutral()`: 用于因子中性化的多元线性回归.
        -- `.stats.standard()`: 高斯或均匀分布的横截面标准化.

    IV  analysis/ (绩效分析)
        策略评估与风险指标.
        -- `.analysis.maxdown()`: 计算详细的最大回撤统计数据.
        -- `.analysis.sharpe()`: 年化夏普比率计算.

    V   db/ & tools/ (实用工具)
        基础设施与数据清洗.
        -- `.db.write()`: 将 DataFrame 直接持久化至 DuckDB/MySQL.
        -- `.tools.log()`: 符号保留的对数变换.

核心接口示例
------------
    I   因子中性化与标准化:
        
        # 对因子进行标准化, 并针对行业和市值进行中性化处理
        factor = df.stats.standard().stats.neutral(ind=ind_df, size=size_df).resid

    II  滚动窗口分析:

        # 获取 21 日滚动窗口内成交量最大的 3 个交易日的平均值
        top_vol = df.rollings(21).max(3).mean()
