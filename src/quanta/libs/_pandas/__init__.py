import pandas as pd
import os
from pathlib import Path
import importlib

import warnings
warnings.simplefilter(action='ignore')
# Third-party library imports
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
path = 'quanta.libs._pandas'
folders = [x.name for x in MODULE_DIR.iterdir() if x.is_dir() and not x.name.startswith('__')]
for folder_name in folders:
    try:
        # 使用 import_module 代替 exec 动态拼接字符串
        module_path = f"{path}.{folder_name}.main"
        module = importlib.import_module(module_path)
        # 如果需要将 main 导出到全局，手动赋值
        globals()[f"{folder_name}_main"] = module
        #print(f"Successfully imported {module_path}")
    except Exception as e:
        print(f"Failed to import {folder_name}: {e}")
