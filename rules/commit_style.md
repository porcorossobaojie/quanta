# Git Commit Style Guide

This document defines the git commit message standards for the `quanta` project. Every commit must be informative, bilingual, and clearly state the scope of changes.

## 1. Commit Structure

A commit message consists of two mandatory parts:
1.  **Subject Line**: A concise English summary of the change.
2.  **Description**: A detailed, bilingual (English | Chinese) breakdown of the updates.

## 2. Formatting Standards

### 2.1 Subject Line
*   **Prefix**: Use standard prefixes like `feat:`, `fix:`, `docs:`, `refactor:`, or `style:`.
*   **Content**: Write a clear, action-oriented English summary.
*   **Example**: `docs: align Barra factor library with contemporary code style standards`

### 2.2 Bilingual Description (CRITICAL)
Every non-trivial commit must provide a mirrored English and Chinese description.
*   **Sectioned Layout**: Use headers like `Key Updates | 主要变更` to separate logical groups.
*   **Point-by-Point Detail**: Specifically list what was changed in which module or file (e.g., specific factors in Barra).
*   **Punctuation**: Use half-width (ASCII) punctuation even in Chinese sections where appropriate.

## 3. Reference Example

**Subject:**
`docs: update Barra factor documentation`

**Description:**
```text
Refactored docstrings in `src/quanta/faclib/barra` to adhere to project standards.
更新 `src/quanta/faclib/barra` 中的文档字符串, 以符合项目标准.

Key Updates | 主要变更:
- Updated Momentum and Residual Volatility factors in `usa4.py`.
  更新了 `usa4.py` 中的动量和残差波动率因子.
- Integrated bilingual docstrings with 79-character separators.
  集成了带有 79 字符分隔符的双语文档字符串.
```

---

# Git 提交风格指南

本文档定义了 `quanta` 项目的 git 提交信息标准. 每次提交必须具有信息性, 采用双语编写, 并明确说明修改范围.

## 1. 提交结构

提交信息由两个强制部分组成:
1.  **主题行 (Subject Line)**: 变更的简明英文摘要.
2.  **描述 (Description)**: 详细的中英双语对照更新说明.

## 2. 格式标准

### 2.1 主题行
*   **前缀**: 使用标准前缀, 如 `feat:`, `fix:`, `docs:`, `refactor:`, 或 `style:`.
*   **内容**: 编写清晰、以动词开头的英文摘要.

### 2.2 双语描述 (至关重要)
每个非平凡的提交必须提供镜像的中英文描述.
*   **分段布局**: 使用 `Key Updates | 主要变更` 等标题分隔逻辑组.
*   **逐点详述**: 具体列出在哪个模块或文件中更改了什么 (例如 Barra 中的特定因子).
*   **标点符号**: 在适当的情况下, 中文部分也应使用半角 (ASCII) 标点符号.
