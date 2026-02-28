from quanta.config import settings as _settings
_config = _settings('data').tables.id_table

from .main import main as _class_obj

__all__ = ['daily']
def daily():
    for i in _config.values():
        instance_obj = _class_obj(**i)
        instance_obj.daily()
        
    
