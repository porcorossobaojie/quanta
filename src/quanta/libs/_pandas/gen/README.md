# Generation & Grouping (gen)

Summary
-------
    The `gen` submodule provides tools for cross-sectional grouping, ranking, 
    and portfolio construction. It is designed to handle factor binning and 
    portfolio return calculations with support for advanced features like 
    hysteresis (to reduce turnover) and rolling weighted averages.

    Accessed via the `.gen` accessor on a Pandas DataFrame.

Key Functions
-------------
    I   Grouping & Ranking
        -- `.gen.group()`: Ranks and bins data into groups based on quantiles 
            or specified boundaries. Supports sequential grouping.
        -- `.gen.cut()`: Selects assets based on rank with a hysteresis 
            mechanism (buffer zones) to stabilize portfolio membership.
        -- `.gen.part_cut()`: Phased version of `cut` to further spread 
            turnover over time.

    II  Portfolio Construction
        -- `.gen.weight()`: Applies asset weights with support for 
            normalization and forward-filling.
        -- `.gen.portfolio()`: Calculates aggregate returns for groups, 
            accounting for rebalancing lags and rolling return windows.
        -- `.gen.roll_weight()`: Computes rolling weighted averages, 
            correctly handling missing values in the window.

Examples
--------
    # Group a factor into 10 deciles
    groups = factor_df.gen.group(rule=np.linspace(0, 1, 11))

    # Calculate portfolio returns with a 1-day lag
    port_rets = groups.gen.portfolio(returns=asset_returns, shift=1)

---

# 生成与分组 (gen)

概要
----
    `gen` 子模块提供了用于横截面分组, 排名和投资组合构建的工具. 它旨在
    处理因子分箱和组合收益计算, 并支持迟滞机制 (以减少换手率) 和滚动
    加权平均等高级功能.

    通过 Pandas DataFrame 上的 `.gen` 访问器进行调用.

核心函数
--------
    I   分组与排名
        -- `.gen.group()`: 根据分位数或指定边界对数据进行排名和分箱. 
            支持顺序分组.
        -- `.gen.cut()`: 基于带有迟滞机制 (缓冲带) 的排名选择资产, 
            以稳定组合成员.
        -- `.gen.part_cut()`: `cut` 的分阶段版本, 以进一步随时间分散
            换手率.

    II  投资组合构建
        -- `.gen.weight()`: 应用资产权重, 支持归一化和前向填充.
        -- `.gen.portfolio()`: 计算各组的聚合收益, 考虑调仓滞后和
            滚动收益窗口.
        -- `.gen.roll_weight()`: 计算滚动加权平均值, 正确处理窗口中
            的缺失值.

示例
----
    # 将因子分为 10 组
    groups = factor_df.gen.group(rule=np.linspace(0, 1, 11))

    # 计算滞后 1 期的组合收益
    port_rets = groups.gen.portfolio(returns=asset_returns, shift=1)
