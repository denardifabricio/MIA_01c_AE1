import logging
from logging.handlers import TimedRotatingFileHandler
import os
import inspect

def log_with_class(logger, level, msg, self=None):
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    class_name = self.__class__.__name__ if self else ''
    logger.log(level, f"[{class_name}.{func_name}] {msg}")



LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
from datetime import datetime
date_str = datetime.now().strftime('%Y-%m-%d')
LOG_FILE = os.path.join(LOG_DIR, f'ui2code.{date_str}.log')

logger = logging.getLogger('ui2code_rag')
logger.setLevel(logging.INFO)

handler = TimedRotatingFileHandler(
    LOG_FILE, when='midnight', interval=1, backupCount=7, encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(module)s.%(funcName)s [%(filename)s:%(lineno)d] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Para usar en otros módulos:
# from src.logging_config import logger


