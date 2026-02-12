import logging


def setup_logger(name):
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s - %(filename)s - %(asctime)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(name)
