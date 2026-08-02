# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:37:57 2026

@author: Porco Rosso
"""

from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import jqdatasdk as jq

from quanta.libs.utils import merge_dicts
from quanta.config import settings, login_info
from quanta.libs.db.main import main as db

config = settings('data')

db.__setting__(database=config.public_keys.minfreq_settings.key.database)

class main(db, type('recommand_settings', (), config.public_keys.minfreq_settings.key)):
    pass
main
