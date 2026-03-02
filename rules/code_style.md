# Project Coding Style Guide

This document defines the contemporary coding standards for the `quanta` project. It reflects the project's shift toward attribute-driven configurations and a distinct bilingual documentation structure.

## 1. Naming Conventions

*   **Core Entry Classes**: Use lowercase `main` for the primary class of a module (e.g., `class main:`).
*   **Static Variables & Attributes**: All static configurations, constants, and class-level variables must be **lowercase** `snake_case`. Uppercase constants are deprecated.
*   **Data Objects**: Use `CapWords` (PascalCase) for auxiliary classes or data containers (e.g., `class NeutralObj:`).
*   **Public Interfaces**: Use lowercase `snake_case` for all public functions and methods.
*   **Internal Protocol Methods**: Use **double underscores** `__name__` for methods intended for internal logic, parameter merging, or system hooks (e.g., `__parameters__`).

## 2. Documentation and Formatting Standards

### 2.1 Scope of Modification (CRITICAL)
*   **Permission Limit**: Modifications are strictly limited to **comments (docstrings/inline)**, **formatting of function signatures (line breaks)**, and **type hints**.
*   **Logic Integrity**: No functional logic, control flow, or algorithm implementation shall be altered under any circumstances.

### 2.2 Three-Section Mirror Docstrings
Every non-trivial function and class must implement a bilingual docstring with the following structure:
1.  **Header (`====`)**: Brief English description.
2.  **English Section**: Detailed `Parameters` and `Returns` using NumPy-style conventions.
3.  **Separator (`----`)**: A mirror Chinese translation of the above content.
4.  **Footer (`----`)**: Closing separator.

**Formatting Rules for Separators:**
*   **Exact Horizontal Length**: The total length of separators (`====`, `----`), including leading indentation spaces, must **exactly equal 79 characters**.
*   **Header Underlines**: The `---` lines under section headers (e.g., `Parameters`, `Returns`) must be **exactly the same length** as the characters above them.

### 2.3 Function Signature Formatting
*   **Argument Line Breaks**: If a function has more than 3 arguments (excluding system defaults like `self`, `cls`, `*args`, `**kwargs`), the signature must use multi-line formatting for better readability.
*   **Mandatory Type Hinting**: All function signatures must include explicit type hints for arguments and return values.

## 3. Engineering Principles

*   **Vectorization First**: For quantitative calculations, prioritize `numpy` and `pandas` vectorized operations over explicit Python loops.
*   **Half-Width Punctuation**: All punctuation in documentation and comments must be half-width (ASCII).

---

# 项目编码风格指南

本文档定义了 `quanta` 项目的现行编码标准. 它反映了项目向属性驱动配置的转变, 以及独特的双语文档结构.

## 1. 命名约定

*   **核心入口类**: 模块的主类使用小写 `main` (例如 `class main:`).
*   **静态变量与属性**: 所有静态配置, 常量和类级变量必须使用**全小写** `snake_case`. 不再使用大写常量.
*   **数据对象**: 辅助类或数据容器类使用 `CapWords` (例如 `class NeutralObj:`).
*   **公共接口**: 所有公共函数和方法使用小写 `snake_case`.
*   **内部协议方法**: 对于内部逻辑, 参数合并或系统钩子的方法, 使用**双下划线** `__name__` (例如 `__parameters__`).

## 2. 文档与格式化标准

### 2.1 修改权限范围 (至关重要)
*   **权限限制**: 修改仅限于**注释 (文档字符串/行内注释)**, **函数签名格式 (断行调整)** 以及**类型提示**.
*   **逻辑完整性**: 在任何情况下均不得改动任何功能逻辑, 控制流或算法实现.

### 2.2 三段镜像文档字符串
每个非平凡的函数和类必须实现具有以下结构的双语文档字符串:
1.  **页眉 (`====`)**: 简要英文描述.
2.  **英文段落**: 使用 NumPy 风格规范的详细 `Parameters` 和 `Returns`.
3.  **分隔符 (`----`)**: 上述内容的镜像中文翻译.
4.  **页脚 (`----`)**: 结束分隔符.

**分隔符格式规则:**
*   **精确横向长度**: 分隔符 (`====`, `----`) 的总长度(包括前导缩进空格)必须**精确等于 79 个字符**.
*   **标题下划线**: 章节标题(如 `Parameters`, `Returns`, `参数`, `返回`)下方的 `---` 长度必须与上方的**字符宽度完全一致**.

### 2.3 函数签名格式化
*   **参数断行**: 如果一个函数的参数超过 3 个(不包括 `self`, `cls`, `*args`, `**kwargs` 等系统默认参数), 必须采用多行格式书写以提高可读性.
*   **强制类型提示**: 所有函数签名必须包含参数和返回值的显式类型提示.

## 3. 工程原则

*   **向量化优先**: 对于量化计算, 优先使用 `numpy` 和 `pandas` 的向量化操作, 而非显式的 Python 循环.
*   **半角标点**: 文档和注释中的所有标点符号(包括中文部分)必须使用半角 (ASCII).
