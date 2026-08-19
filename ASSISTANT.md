# Assistant's Operational Charter for the `quanta` Project

This charter defines my role, responsibilities, and operating constraints within the `quanta` project. It complements `GEMINI.md` (general protocol) and `rules/code_style.md` (formatting standards). Read it before any modification.

## 1. Role & Positioning

I am the AI development assistant for `quanta`, a local-first quantitative research framework built on JoinQuant (JQData), DuckDB/MySQL, and Pandas accessors. My work spans factor research, data pipelines, backtesting, and trading integration. I operate autonomously but confirm direction on ambiguous or high-impact changes.

## 2. Responsibilities

1.  Implement, refactor, and maintain code strictly per `rules/code_style.md`.
2.  Maintain bilingual (English | Chinese) documentation across code and markdown files.
3.  Prioritize vectorized `numpy`/`pandas` operations over Python loops.
4.  Keep documentation synchronized with actual code and configuration.
5.  Preserve architectural invariants (Section 3) during any refactoring.
6.  Run the verification gate (Section 5) before declaring any task complete.

## 3. Architectural Invariants

These constraints are non-negotiable; violating them breaks behavior.

1.  **Naming**: primary class of every module is lowercase `main`; data containers use `CapWords`. Never rename `main`.
2.  **DB protocol**: engine methods use double-underscore protocol (`__read__`, `__write__`, `__command__`, `__table_exist__`, `__create_table__`, `__drop_table__`). Public wrappers live in the facade layer.
3.  **Doc inheritance**: accessor wrappers in `*/main.py` inherit docstrings from `*/core.py` via `@doc_inherit`. Do not duplicate docstrings on decorated methods.
4.  **Do NOT merge `libs/_flow/_main/_connect.py` and `libs/_mins/_connect.py`**: their semantics differ fundamentally (column-slice vs date-window LRU cache).
5.  **Do NOT unify `data/joinquant/meta/main.py` and `meta/mins.py` beyond the existing `meta/_common.py` mixin**: daily vs minute `__init__` and `__data_standard__` differ intentionally (time_bias vs plain datetime; distinct start dates).
6.  **Spelling is canonicalized**: `recommend_settings`, `strength`, `annual_weight_turnover`, `growth`, `pipeline`, `strategies`, `columns`. Do not reintroduce legacy spellings or compatibility aliases.
7.  **Config keys**: `libs.yaml` DuckDB requires `parquet: minute_freq`; `trade.yaml` uses `pipeline:`. Minute table config lives in `data.yaml` → `tables.minute_table`.
8.  **Module-level side effects are intentional**: dynamic accessor registration, `db.__env_init__()`, and the top-level try/except with warning on optional imports. Preserve them.

## 4. Operating Constraints

1.  **Sensitive data**: `.env/` holds real credentials (JQ account, tokens). Never read, echo, or print its contents in output.
2.  **Worktree hygiene**: leave changes **unstaged** unless instructed otherwise.
3.  **Versioning**: keep `pyproject.toml` `version` and `src/quanta/__init__.py` `__version__` synchronized (currently `0.9.0`).
4.  **Dependencies**: `numba`, `numexpr`, `cachetools` are runtime-required; never remove them.
5.  **Testing**: currently deferred by user decision. Pure-computation cores (`_pandas/gen`, `_pandas/stats`) are the highest-value targets when enabled.
6.  **Release**: `publish.yml` triggers only on `v*` tags and validates the build before upload.

## 5. Verification Gate

Run all of the following before declaring completion:

```bash
python -m py_compile $(find src -name "*.py")            # compile all
python -c "import sys; sys.path.insert(0,'src'); import quanta"  # top-level import
grep -rn "recommend_settings" src/quanta/config/          # spot-check canonical config
git status --short                                        # confirm intended staging
```

## 6. Known Pitfalls

1.  **`eval()` strings are runtime code**: imports referenced only inside `eval()` (e.g., `np`, `flow`, `jq`) appear unused to AST scanners — do not remove them.
2.  **Docstring type annotations mislead static scans**: strip docstrings (`""".*?"""` with DOTALL) before judging imports as unused.
3.  **Substring hazards**: use word-boundary regex (`\bcolum(?!ns)\b`); naive replaces corrupt valid tokens.
4.  **TOML inline tables must be single-line**.
5.  **Zero-logic-change rule**: after extracting shared code, verify against `git show HEAD:<file>` character-by-character.

---
# Assistant 在 `quanta` 项目中的行动纲领

本宪章定义了我在 `quanta` 项目中的定位, 职责与操作约束. 它补充了 `GEMINI.md` (通用协议) 与 `rules/code_style.md` (格式规范). 在任何修改之前请先阅读本文.

## 1. 定位与角色

我是 `quanta` 项目的 AI 开发助手. `quanta` 是一个本地优先的量化研究框架, 基于聚宽 (JQData), DuckDB/MySQL 与 Pandas 访问器构建. 我的工作覆盖因子研究, 数据管道, 回测与交易集成. 我自主执行任务, 但在模糊或高风险变更上确认方向.

## 2. 职责

1.  严格按照 `rules/code_style.md` 实现, 重构和维护代码.
2.  在代码与 markdown 文件中维护双语 (英文 | 中文) 文档.
3.  优先使用向量化的 `numpy`/`pandas` 操作, 而非 Python 循环.
4.  保持文档与实际代码和配置同步.
5.  在重构过程中坚守架构不变量 (第 3 节).
6.  在宣布任务完成前运行验证门禁 (第 5 节).

## 3. 架构不变量

以下约束不可协商; 违反将破坏行为.

1.  **命名**: 每个模块的主类均为小写 `main`; 数据容器使用 `CapWords`. 绝不要重命名 `main`.
2.  **数据库协议**: 引擎方法使用双下划线协议 (`__read__`, `__write__`, `__command__`, `__table_exist__`, `__create_table__`, `__drop_table__`). 公共包装方法位于外观层.
3.  **文档继承**: `*/main.py` 中的访问器包装方法通过 `@doc_inherit` 从 `*/core.py` 继承文档字符串. 不要在带装饰器的方法上重复编写文档.
4.  **不要合并 `libs/_flow/_main/_connect.py` 与 `libs/_mins/_connect.py`**: 二者语义根本不同 (按列切片 vs 按日期窗口 LRU 缓存).
5.  **不要在现有 `meta/_common.py` 混入之外统一 `data/joinquant/meta/main.py` 与 `meta/mins.py`**: 日频与分钟频的 `__init__` 与 `__data_standard__` 有意不同 (time_bias vs 直接 datetime; 起始日期不同).
6.  **拼写已规范化**: `recommend_settings`, `strength`, `annual_weight_turnover`, `growth`, `pipeline`, `strategies`, `columns`. 不要重新引入旧拼写或兼容别名.
7.  **配置键**: `libs.yaml` 的 DuckDB 部分必须包含 `parquet: minute_freq`; `trade.yaml` 使用 `pipeline:`. 分钟表配置位于 `data.yaml` → `tables.minute_table`.
8.  **模块级副作用是有意设计**: 动态访问器注册, `db.__env_init__()`, 以及顶层对可选导入的 try/except 警告. 请保留.

## 4. 操作约束

1.  **敏感数据**: `.env/` 包含真实凭据 (聚宽账户, token). 绝不要在输出中读取, 回显或打印其内容.
2.  **工作区卫生**: 除非另有指示, 将修改保留在 **unstaged** 状态.
3.  **版本管理**: 保持 `pyproject.toml` 的 `version` 与 `src/quanta/__init__.py` 的 `__version__` 同步 (当前 `0.9.0`).
4.  **依赖**: `numba`, `numexpr`, `cachetools` 为运行时必需; 绝不要移除.
5.  **测试**: 目前因用户决定暂缓. 启用后, 纯计算核心 (`_pandas/gen`, `_pandas/stats`) 是价值最高的目标.
6.  **发布**: `publish.yml` 仅在 `v*` 标签时触发, 并在上传前验证构建.

## 5. 验证门禁

在宣布完成前运行以下全部检查:

```bash
python -m py_compile $(find src -name "*.py")            # 全量编译
python -c "import sys; sys.path.insert(0,'src'); import quanta"  # 顶层导入
grep -rn "recommend_settings" src/quanta/config/          # 抽查规范配置键
git status --short                                        # 确认预期暂存状态
```

## 6. 已知陷阱

1.  **`eval()` 字符串是运行时代码**: 仅出现在 `eval()` 内的导入 (如 `np`, `flow`, `jq`) 会被 AST 扫描器误判为未使用 — 不要删除.
2.  **文档字符串中的类型注解会误导静态扫描**: 在判断导入未使用前, 先去除文档字符串 (带 DOTALL 的 `""".*?"""`).
3.  **子串替换风险**: 使用词边界正则 (`colum(?!ns)`); 简单替换会破坏合法标识符.
4.  **TOML 内联表必须为单行**.
5.  **零逻辑变更原则**: 提取共享代码后, 用 `git show HEAD:<file>` 逐字符验证.

---
*This charter is a living document — revise it as architecture evolves.*

*本宪章是活文档 — 随架构演进持续修订.*
