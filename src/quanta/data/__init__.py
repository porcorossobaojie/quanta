from .joinquant import daily as _jq_daily, minute as _jq_minute

def daily() -> None:
    """Builds the local daily database | 构建本地日频数据库"""
    _jq_daily()
    
def minute() -> None:
    """Builds the local minute database | 构建本地分钟频数据库"""
    _jq_minute()
    
