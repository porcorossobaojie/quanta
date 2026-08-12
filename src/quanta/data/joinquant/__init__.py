
from .dt_table import daily as dt_daily
from .id_table import daily as id_daily
from .mt_table import daily as _minute_daily

def daily() -> None:
    """Builds the local daily database from JoinQuant | 从聚宽构建本地日频数据库"""
    import jqdatasdk as jq
    from quanta.config import login_info
    jq.auth(**login_info('account').joinquant)
    id_daily()
    dt_daily()

def minute() -> None:
    """Builds the local minute database from JoinQuant | 从聚宽构建本地分钟频数据库"""
    import jqdatasdk as jq
    from quanta.config import login_info
    jq.auth(**login_info('account').joinquant)
    _minute_daily()
