# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:29:13 2026

@author: Porco Rosso
"""

import numpy as np
import pandas as pd
from functools import lru_cache

def rsi(ret, periods):
    ag = ret[ret > 0].rolling(periods).sum()
    al = ret[ret <= 0].rolling(periods).sum().replace(0, np.nan)
    rs = ag / al
    rsi = 100 - (100 / (1 + rs.fillna(0))).f.tradestatus()
    return rsi


    
