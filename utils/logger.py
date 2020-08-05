import logging

FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'

def setup_logger(log_file):
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format=FMT, datefmt=DATEFMT)

def get_file_handler(log_file, level=10):
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_format = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
    file_handler.setFormatter(file_format)
    return file_handler

def get_logger(name, log_file=None, level=10):
    log_file = log_file if log_file else name + '.log'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(get_file_handler(log_file, level))
    logger.propagate = False
    return logger
