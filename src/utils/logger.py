import logging
from src.config import Config


def setup_logger(name):
    log_level = logging.INFO if not Config.DEBUG else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s - %(filename)s:%(lineno)d - %(asctime)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(name)
