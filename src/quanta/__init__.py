from .libs import _pandas

try:
    from .libs import _flow as flow
    from .libs import _mins as mins
    from .trade.account.main import main as account
    from .strategies import meta as strategies
    from . import config, libs, data
    from . import faclib
except Exception as e:
    print(f"[quanta] optional imports partially failed: {e}")

__version__ = "0.9.0"
