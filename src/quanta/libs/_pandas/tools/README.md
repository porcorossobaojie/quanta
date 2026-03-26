# Utility Tools (tools)

Summary
-------
    The `tools` submodule provides specialized utility functions for data 
    cleaning, transformation, and low-level array manipulation. It includes 
    efficient implementations for forward-filling sparse data, sign-preserved 
    logarithmic transformations, and high-performance rolling window creation 
    using NumPy strides.

    Accessed via the `.tools` accessor on Pandas DataFrame or Series.

Key Functions
-------------
    I   Data Cleaning & Filling
        -- `.tools.fillna()`: Efficiently forward-fills a DataFrame to match 
            a new index, extending the dataset as needed.
        -- `.tools.shift()`: Iteratively shifts data down to fill specific 
            terminal NaN values.

    II  Transformations & Array Utilities
        -- `.tools.log()`: Sign-preserved or standard log transformation, 
            ideal for handling financial data with both positive and negative 
            values.
        -- `.tools.halflife()`: Generates exponential decay weights based 
            on a given half-life.
        -- `.tools.array_roll()`: Creates a high-performance rolling window 
            view of a NumPy array using stride tricks, returning a 3D structure.

Examples
--------
    # Apply sign-preserved log to handle negative returns
    log_rets = returns_df.tools.log()

    # Extend a factor DataFrame to match a list of trading dates
    full_df = sparse_df.tools.fillna(fill_list=trading_dates)

    # Generate decay weights for a 20-day window
    weights = pd.tools.halflife(20, 10)

---

# 实用工具 (tools)

概要
----
    `tools` 子模块提供了用于数据清洗, 变换和底层数组操作的专用实用函数. 
    它包括高效的稀疏数据前向填充, 符号保留的对数变换以及使用 NumPy 步长
    的高性能滚动窗口创建.

    通过 Pandas DataFrame 或 Series 上的 `.tools` 访问器进行调用.

核心函数
--------
    I   数据清洗与填充
        -- `.tools.fillna()`: 高效地前向填充 DataFrame 以匹配新索引, 
            并根据需要扩展数据集.
        -- `.tools.shift()`: 迭代向下移动数据以填充特定的末端 NaN 值.

    II  变换与数组工具
        -- `.tools.log()`: 符号保留或标准的对数变换, 非常适合处理同时
            具有正值和负值的金融数据.
        -- `.tools.halflife()`: 根据给定的半衰期生成指数衰减权重.
        -- `.tools.array_roll()`: 使用步长技巧从 NumPy 数组创建高性能
            滚动窗口视图, 返回 3D 结构.

示例
----
    # 应用符号保留的对数处理以应对负收益率
    log_rets = returns_df.tools.log()

    # 扩展因子 DataFrame 以匹配交易日期列表
    full_df = sparse_df.tools.fillna(fill_list=trading_dates)

    # 为 20 日窗口生成衰减权重
    weights = pd.tools.halflife(20, 10)
