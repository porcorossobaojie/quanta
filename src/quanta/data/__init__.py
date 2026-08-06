from .joinquant import daily as _jq_daily, minute_daily as _minute_daily

def daily():
    _jq_daily()
    
def minute():
    _minute_daily()
    
