import logging


def setup_logger(log_file):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format=log_format)

def get_file_handler(log_file, level=10):
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = logging.Formatter(log_format)
    file_handler.setFormatter(file_format)
    return file_handler

def get_logger(name, log_file=None, level=10):
    log_file = log_file if log_file else name + '.log'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(get_file_handler(log_file, level))
    logger.propagate = False
    return logger
