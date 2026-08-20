import logging
from easy_to_markdown.pkg import DISABLE_AUTO_LOGGING_CONFIG

logger = logging.getLogger("easy_to_markdown")


def _set_up_logger():
    if DISABLE_AUTO_LOGGING_CONFIG:
        return

    # Basically compatible with PaddleOCR 2.x, except for logging to stderr
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False


_set_up_logger()
