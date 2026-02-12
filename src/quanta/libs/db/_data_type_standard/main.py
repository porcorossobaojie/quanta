# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:38:54 2026

@author: Porco Rosso
"""

from src.quanta.config import settings
config = settings('libs').db

__all__ = ['data_trans']

def data_trans(data_type, recommand_settings=None):
    parts = data_type.split('(')
    base_type = parts[0].upper()
    recommand_settings = config.recommand_settings if recommand_settings is None else recommand_settings
    dic = config[recommand_settings].data_type.to_dict()
    translated_type = dic.get(base_type, dic['UNDIFINED'])
    if len(parts) > 1 and (translated_type in dic['MPS']):
        return f"{translated_type}({parts[1]}"
    return translated_type
    
