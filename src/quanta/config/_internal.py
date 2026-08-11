# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:22:33 2026

@author: Porco Rosso
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv # Import dotenv functions
from box import Box
import yaml
from typing import Optional, List

# Load environment variables from .env file, searching from the current working directory
load_dotenv(find_dotenv(usecwd=True))

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_project_root_containing_env_folder() -> Path:
    """Finds the project root containing the '.env' folder | 查找包含 '.env' 文件夹的项目根目录"""
    current_dir = Path(os.getcwd())
    while True:
        if (current_dir / '.env').is_dir():
            return current_dir

        # Stop if we reach the filesystem root or the drive root
        if current_dir == current_dir.parent:
            break

        current_dir = current_dir.parent

    # Fallback: if no '.env' folder found, use the current working directory
    return Path(os.getcwd())

PROJECT_ROOT = _find_project_root_containing_env_folder()

def _yaml_config(files: List[Path]) -> Box:
    """Loads and merges YAML files into a Box config | 加载并合并 YAML 文件为 Box 配置"""
    config = Box(default_box=False, box_dots=True)
    for i in files:
        with open(str(i), 'r', encoding = 'utf-8') as f:
            x = yaml.safe_load_all(f)
            for j in x:
                if j:
                    config.merge_update(j)
    return config

__all__ = ['settings', 'login_info']

def settings(
    yaml_file: str,
    env_file: Optional[str] = None
) -> Box:
    """
    ===========================================================================
    Loads configuration from default and override YAML files.

    Parameters
    ----------
    yaml_file : str
        The base configuration file name (with or without '.yaml').
    env_file : Optional[str]
        The override file name in the project's '.env' folder.
        Default is None (same as yaml_file).

    Returns
    -------
    Box
        The merged configuration object.
    ---------------------------------------------------------------------------
    从默认和覆盖 YAML 文件加载配置.

    参数
    ----
    yaml_file : str
        基础配置文件名 (可带或不带 '.yaml').
    env_file : Optional[str]
        项目 '.env' 文件夹中的覆盖文件名. 默认为 None (与 yaml_file 相同).

    返回
    ----
    Box
        合并后的配置对象.
    ---------------------------------------------------------------------------
    """
    if yaml_file[-5:].lower() != '.yaml':
        yaml_file = f"{yaml_file}.yaml"

    config_files = []

    # 1. Add default config file from quanta package
    default_config_path = Path(MODULE_DIR) / yaml_file
    if default_config_path.is_file():
        config_files.append(default_config_path)

    # 2. Add override config file from project's .env folder
    override_filename = yaml_file if env_file is None else env_file
    override_config_path = PROJECT_ROOT / '.env' / override_filename
    if override_config_path.is_file():
        config_files.append(override_config_path)

    if not config_files:
        raise FileNotFoundError(f"No configuration files found for '{yaml_file}' in quanta or project's .env folder.")

    base = _yaml_config(config_files)
    return base

def login_info(
    env_file: str
) -> Box:
    """
    ===========================================================================
    Loads login credentials from the project's '.env' folder.

    Parameters
    ----------
    env_file : str
        The login info file name (with or without '.yaml').

    Returns
    -------
    Box
        The merged login credential configuration.
    ---------------------------------------------------------------------------
    从项目的 '.env' 文件夹加载登录凭据.

    参数
    ----
    env_file : str
        登录信息文件名 (可带或不带 '.yaml').

    返回
    ----
    Box
        合并后的登录凭据配置.
    ---------------------------------------------------------------------------
    """
    if env_file[-5:].lower() != '.yaml':
        env_file = f"{env_file}.yaml"

    config_files = []

    # This function seems specifically designed to load from the project's .env folder
    # However, if there's a default login_info.yaml in quanta/config, we should include it first.
    # For now, let's assume login_info only comes from the project's .env folder as per original intent.
    # If there's a default, it would be Path(MODULE_DIR) / env_file

    override_config_path = PROJECT_ROOT / '.env' / env_file
    if override_config_path.is_file():
        config_files.append(override_config_path)

    if not config_files:
        raise FileNotFoundError(f"Login info file '{env_file}' not found in project's .env folder.")

    base = _yaml_config(config_files)
    return base
