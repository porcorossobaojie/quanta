
from .dt_table import daily as dt_daily
from .id_table import daily as id_daily

def daily():
    import jqdatasdk as jq
    from quanta.config import login_info
    jq.auth(**login_info('account').joinquant)
    id_daily()
    dt_daily()
