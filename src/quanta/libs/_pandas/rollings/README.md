# Rolling Window Operations (rollings)

Summary
-------
    The `rollings` submodule provides high-performance rolling window 
    operations optimized for time-series data. Beyond standard rolling 
    means or sums, it supports complex operations like retrieving the top 
    N values in a window, rolling time-series ranking, and custom function 
    application over windowed data.

    Accessed via the `.rollings(window)` accessor on Pandas DataFrame or Series.

Key Functions
-------------
    I   Ordered Window Statistics
        -- `.rollings(w).max(n).mean()`: Retrieves the mean of the largest 
            N values within each rolling window of size W.
        -- `.rollings(w).min(n).sum()`: Retrieves the sum of the smallest 
            N values within each rolling window.
        -- `.rollings(w).max().apply(func)`: Applies a custom function to 
            the N largest values in the window.

    II  Time-Series Ranking
        -- `.rollings(w).ts_rank()`: Calculates the percentage rank of the 
            current value relative to the past W observations.

Examples
--------
    # Get the average of the 3 highest returns over a 20-day window
    top_rets = df.rollings(20).max(3).mean()

    # Calculate 252-day time-series rank (momentum-like)
    ts_rank = df.rollings(252).ts_rank()

---

# 滚动窗口操作 (rollings)

概要
----
    `rollings` 子模块提供了针对时间序列数据优化的高性能滚动窗口操作. 除
    了标准的滚动均值或求和外, 它还支持复杂的运算, 如检索窗口内的前 N 个
    值, 滚动时间序列排名以及在窗口化数据上应用自定义函数.

    通过 Pandas DataFrame 或 Series 上的 `.rollings(window)` 访问器
    进行调用.

核心函数
--------
    I   有序窗口统计
        -- `.rollings(w).max(n).mean()`: 在大小为 W 的每个滚动窗口内, 
            获取最大的 N 个值的平均值.
        -- `.rollings(w).min(n).sum()`: 获取每个滚动窗口内最小的 N 个
            值的和.
        -- `.rollings(w).max().apply(func)`: 将自定义函数应用于窗口内
            最大的 N 个值.

    II  时间序列排名
        -- `.rollings(w).ts_rank()`: 计算当前值相对于过去 W 个观测值
            的百分比排名.

示例
----
    # 获取 20 日窗口内最高的 3 个收益率的平均值
    top_rets = df.rollings(20).max(3).mean()

    # 计算 252 日时间序列排名 (类似于动量)
    ts_rank = df.rollings(252).ts_rank()
