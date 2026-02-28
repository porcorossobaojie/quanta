from quanta.config import settings as _settings
_portfolio_types = _settings('data').public_keys.recommand_settings.portfolio_types

from ._base import main as _meta
from ._connect import calendar_days, trade_days

__instance__ = {}

for _i in _portfolio_types:
    __instance__[_i] = _meta(_i)
    globals()[_i] = __instance__[_i]


__all__ = _portfolio_types.to_list() + ['calendar_days', 'trade_days']
__instance__.update({'calendar_days':calendar_days, 'trade_days': trade_days})
