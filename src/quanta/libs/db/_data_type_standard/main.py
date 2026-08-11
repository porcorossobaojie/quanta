# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:38:54 2026

@author: Porco Rosso
"""

from ....config import settings
from typing import Optional
config = settings('libs').db

__all__ = ['data_trans']

def data_trans(
    data_type: str,
    recommand_settings: Optional[str] = None
) -> str:
    """
    ===========================================================================
    Translates a data type string to the target engine standard.

    Parameters
    ----------
    data_type : str
        The source data type, possibly with a size argument (e.g., 'VARCHAR(10)').
    recommand_settings : Optional[str]
        The target engine name. Default is None (configured engine).

    Returns
    -------
    str
        The translated data type in the target standard.
    ---------------------------------------------------------------------------
    将数据类型字符串转换为目标引擎标准.

    参数
    ----
    data_type : str
        源数据类型, 可能带大小参数 (例如 'VARCHAR(10)').
    recommand_settings : Optional[str]
        目标引擎名称. 默认为 None (配置的引擎).

    返回
    ----
    str
        目标标准下的数据类型.
    ---------------------------------------------------------------------------
    """
    parts = data_type.split('(')
    base_type = parts[0].upper()
    recommand_settings = config.recommand_settings if recommand_settings is None else recommand_settings
    dic = config[recommand_settings].data_type.to_dict()
    translated_type = dic.get(base_type, dic['UNDIFINED'])
    if len(parts) > 1 and (translated_type in dic['MPS']):
        return f"{translated_type}({parts[1]}"
    return translated_type
    
