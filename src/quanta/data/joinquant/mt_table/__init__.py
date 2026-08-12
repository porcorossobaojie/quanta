from ....config import settings as _settings
_config = _settings('data').tables.minute_table

from .main import main as _class_obj

__all__ = ['daily']
def daily() -> None:
    """Runs the daily ETL pipeline for all configured tables | 运行所有配置表的日频 ETL 管道"""
    for i in _config.values():
        instance_obj = _class_obj(**i)
        instance_obj.daily()
