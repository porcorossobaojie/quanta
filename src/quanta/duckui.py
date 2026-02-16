# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 17:56:33 2026

@author: Porco Rosso
"""

import subprocess
from quanta.libs.db.main import main as DuckDB

cmd = [
    f"{DuckDB.path}/duckdb.exe",
    f"{DuckDB.path}/{DuckDB.database}.duckdb",
#    "-readonly",
    "-ui"
]

# 启动 DuckDB UI（会阻塞当前 Python 进程，直到 UI 退出）
subprocess.run(cmd)
