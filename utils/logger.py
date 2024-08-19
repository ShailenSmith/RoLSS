import logging
import sys

FORMATTER = logging.Formatter("%(asctime)s : %(levelname)-5s : %(message)s")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler(filename):
    file_handler = logging.FileHandler( f"{filename}.log", 'w' )
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name, filename):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(filename))
    logger.propagate = False
    return logger