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

# Load environment variables from .env file, searching from the current working directory
load_dotenv(find_dotenv(usecwd=True))

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_project_root_containing_env_folder():
    """
    Finds the project root by searching upwards from the current working directory
    for a directory that contains a '.env' folder.
    """
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

def _yaml_config(files):
    config = Box(default_box=False)
    for i in files:
        with open(str(i), 'r', encoding = 'utf-8') as f:
            x = yaml.safe_load_all(f)
            for j in x:
                if j:
                    config.merge_update(j)
    return config

__all__ = ['settings', 'login_info']

def settings(yaml_file, env_file=None):
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

def login_info(env_file):
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
