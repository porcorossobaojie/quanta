# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:22:33 2026

@author: Porco Rosso
"""

import os
from pathlib import Path
from src.quanta.libs.utils import yaml_config as _yaml_config

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ['settings', 'login_info']

def settings(yaml_file, env_file=None):
    if yaml_file[-5:].lower() != '.yaml':
        yaml_file = f"{yaml_file}.yaml"
    try:
        base = _yaml_config(
            [Path(MODULE_DIR) / yaml_file,
             Path(MODULE_DIR).parents[2] / '.env' / (yaml_file if env_file is None else env_file)]
        )
    except:
        base = _yaml_config(
            [Path(MODULE_DIR) / yaml_file]
        )
    return base

def login_info(env_file):
    if env_file[-5:].lower() != '.yaml':
        env_file = f"{env_file}.yaml"
    base = _yaml_config([Path(MODULE_DIR).parents[2] / '.env' / env_file])
    return base
    
