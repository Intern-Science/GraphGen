import logging

logger = logging.getLogger("graphgen")

def set_logger(log_file: str, log_level: int = logging.INFO, if_stream: bool = True):
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    stream_handler = None

    if if_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        if if_stream and stream_handler:
            logger.addHandler(stream_handler)


def parse_log(log_file: str):
    with open(log_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    return lines
