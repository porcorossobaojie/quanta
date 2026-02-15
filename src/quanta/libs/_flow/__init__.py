from quanta.config import settings as _settings
_portfolio_types = _settings('data').public_keys.recommand_settings.portfolio_types

from .main import main as _meta
from ._connect import calendar_days, trade_days

for _i in _portfolio_types:
    globals()[_i] = _meta(_i)

__all__ = _portfolio_types.to_list() + ['calendar_days', 'trade_days']



