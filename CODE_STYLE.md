# Project Coding Style Guide

This document defines the contemporary coding standards for the `quanta` project. It reflects the project's shift toward attribute-driven configurations and a distinct bilingual documentation structure.

## 1. Naming Conventions

*   **Core Entry Classes**: Use lowercase `main` for the primary class of a module (e.g., `class main:`).
*   **Static Variables & Attributes**: All static configurations, constants, and class-level variables must be **lowercase** `snake_case`. Uppercase constants are deprecated.
*   **Data Objects**: Use `CapWords` (PascalCase) for auxiliary classes or data containers (e.g., `class NeutralObj:`).
*   **Public Interfaces**: Use lowercase `snake_case` for all public functions and methods.
*   **Internal Protocol Methods**: Use **double underscores** `__name__` for methods intended for internal logic, parameter merging, or system hooks (e.g., `__parameters__`). This clearly distinguishes internal system protocols from public APIs.

## 2. Documentation Standards

### 2.1 Three-Section Mirror Docstrings
Every non-trivial function and class must implement a bilingual docstring with the following structure:
1.  **Header (`====`)**: Brief English description.
2.  **English Section**: Detailed `Parameters` and `Returns` using NumPy-style conventions.
3.  **Separator (`----`)**: A mirror Chinese translation of the above content.
4.  **Footer (`----`)**: Closing separator.

**Formatting Rules for Separators:**
*   **Horizontal Length**: The total length of separators (`====`, `----`), including leading indentation spaces, must **not exceed 79 characters**.
*   **Header Underlines**: The `---` lines under section headers (e.g., `Parameters`, `Returns`) must be **exactly the same length** as the characters above them.

### 2.2 Exemption for Simple Functions
Utility or self-explanatory functions (e.g., single-line return, basic wrappers, simple getters) may use simplified single-line docstrings or skip the full three-section format to maintain code brevity.

## 3. Engineering Principles

*   **Mandatory Type Hinting**: All function signatures must include explicit type hints for arguments and return values.
*   **Vectorization First**: For quantitative calculations, prioritize `numpy` and `pandas` vectorized operations over explicit Python loops to ensure performance.
*   **Half-Width Punctuation**: All punctuation in documentation and comments must be half-width (ASCII).

## 4. Code Example

```python
class main:
    def __init__(self, data_source: str = 'local') -> None:
        # Simple method: Exemption applied
        self.source_type: str = data_source

    def calculate_factor(
        self,
        df_obj: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """
        =======================================================================
        Calculates the momentum factor for a given period.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input price DataFrame.
        period : int, optional
            The lookback window, by default 20.

        Returns
        -------
        pd.Series
            The calculated factor values.
        -----------------------------------------------------------------------
        计算给定周期内的动量因子.

        参数
        ----
        df_obj : pd.DataFrame
            输入的证券价格数据.
        period : int, optional
            回溯窗口大小, 默认为 20.

        返回
        ----
        pd.Series
            计算出的因子值.
        -----------------------------------------------------------------------
        """
        return df_obj.pct_change(period).mean()
```

---

# 项目编码风格指南

本文档定义了 `quanta` 项目的现行编码标准. 它反映了项目向属性驱动配置的转变, 以及独特的双语文档结构.

## 1. 命名约定

*   **核心入口类**: 模块的主类使用小写 `main` (例如 `class main:`).
*   **静态变量与属性**: 所有静态配置, 常量和类级变量必须使用**全小写** `snake_case`. 不再使用大写常量.
*   **数据对象**: 辅助类或数据容器类使用 `CapWords` (例如 `class NeutralObj:`).
*   **公共接口**: 所有公共函数和方法使用小写 `snake_case`.
*   **内部协议方法**: 对于内部逻辑, 参数合并或系统钩子的方法, 使用**双下划线** `__name__` (例如 `__parameters__`). 这有助于清晰区分内部系统协议与公共 API.

## 2. 文档标准

### 2.1 三段镜像文档字符串
每个非平凡的函数和类必须实现具有以下结构的双语文档字符串:
1.  **页眉 (`====`)**: 简要英文描述.
2.  **英文段落**: 使用 NumPy 风格规范的详细 `Parameters` 和 `Returns`.
3.  **分隔符 (`----`)**: 上述内容的镜像中文翻译.
4.  **页脚 (`----`)**: 结束分隔符.

**分隔符格式规则:**
*   **横向长度**: 分隔符 (`====`, `----`) 的总长度(包括前导缩进空格)不得超过 **79 个字符**.
*   **标题下划线**: 章节标题(如 `Parameters`, `Returns`, `参数`, `返回`)下方的 `---` 长度必须与上方的**字符宽度完全一致**.

### 2.2 简单函数豁免
对于功能直观, 逻辑简单 (如单行返回, 基础封装, 简单 Getter) 的工具函数, 可以使用单行简化的文档字符串或跳过完整的三段式格式, 以保持代码简洁.

## 3. 工程原则

*   **强制类型提示**: 所有函数签名必须包含参数和返回值的显式类型提示.
*   **向量化优先**: 对于量化计算, 优先使用 `numpy` 和 `pandas` 的向量化操作, 而非显式的 Python 循环, 以确保性能.
*   **半角标点**: 文档和注释中的所有标点符号(包括中文部分)必须使用半角 (ASCII).

## 4. 代码示例

```python
class main:
    def __init__(self, data_source: str = 'local') -> None:
        # 简单方法: 适用豁免规则
        self.source_type: str = data_source

    def calculate_factor(
        self,
        df_obj: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """
        =======================================================================
        Calculates the momentum factor for a given period.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The input price DataFrame.
        period : int, optional
            The lookback window, by default 20.

        Returns
        -------
        pd.Series
            The calculated factor values.
        -----------------------------------------------------------------------
        计算给定周期内的动量因子.

        参数
        ----
        df_obj : pd.DataFrame
            输入的证券价格数据.
        period : int, optional
            回溯窗口大小, 默认为 20.

        返回
        ----
        pd.Series
            计算出的因子值.
        -----------------------------------------------------------------------
        """
        return df_obj.pct_change(period).mean()
```
