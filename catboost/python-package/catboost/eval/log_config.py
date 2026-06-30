import logging

filename = None
eval_logger = None
console_handler = None


def init():
    global eval_logger
    eval_logger_name = "eval_feature"
    eval_logger = logging.getLogger(eval_logger_name)
    eval_logger.setLevel(logging.CRITICAL)
    global console_handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)

    formatter = logging.Formatter("[%(levelname)s]: %(message)s")
    console_handler.setFormatter(formatter)

    eval_logger.addHandler(console_handler)


def set_logger_name(name):
    global filename
    filename = name


def get_eval_logger():
    if eval_logger is None:
        init()
    return eval_logger


def set_level(level):
    if eval_logger is None:
        init()
    console_handler.setLevel(level)
