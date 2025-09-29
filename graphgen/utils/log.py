import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

logger = logging.getLogger("graphgen")


def set_logger(
    log_file: str,
    log_level: int = logging.INFO,
    *,
    if_stream: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 5,
    force: bool = False,
):

    if logger.hasHandlers() and not force:
        return

    if force:
        logger.handlers.clear()

    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if if_stream:
        console = RichHandler(level=log_level, show_path=False, rich_tracebacks=True)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)


def parse_log(log_file: str):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines
