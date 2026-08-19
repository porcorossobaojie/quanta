# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 17:56:33 2026

@author: Porco Rosso
"""
import os
import subprocess
from pathlib import Path

from quanta.libs.db.main import main as DuckDB


def launch_ui() -> None:
    """Launches the DuckDB UI for the configured local database | 为配置的本地数据库启动 DuckDB UI"""
    path = getattr(DuckDB, 'path', None)
    if not path:
        print("[quanta] DuckDB path is not configured; set it in <libs.yaml>.")
        return

    database = getattr(DuckDB, 'database', 'Locals')
    duckdb_exe = Path(path) / 'duckdb.exe'
    db_file = Path(path) / f"{database}.duckdb"

    if not duckdb_exe.is_file():
        print(f"[quanta] duckdb.exe not found at <{duckdb_exe}>.")
        return
    if not db_file.is_file():
        print(f"[quanta] database file not found at <{db_file}>.")
        return

    cmd = [
        str(duckdb_exe),
        str(db_file),
        # "-readonly",
        "-ui"
    ]

    # 启动 DuckDB UI（会阻塞当前 Python 进程，直到 UI 退出）
    subprocess.run(cmd, env={**os.environ})


if __name__ == "__main__":
    launch_ui()
