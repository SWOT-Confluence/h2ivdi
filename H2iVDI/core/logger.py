import io
import logging
import sys

try:
    import colorlog
except:
    colorlog = None


def create_logger(debug_level=0, log_file=None):
    """Create logger"""

    if colorlog is not None:

        # Create a formatter with color support
        formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)-8s]%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "bold_yellow",
                "ERROR": "bold_red",
                "CRITICAL": "bold_purple",
            },
        )

    else:

        # Create a formatter without color support
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')


    # Create handlers
    if log_file is None:

        # Create console handler and set formatter
        if not type(sys.stdout) == io.TextIOWrapper:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    else:

        # Create file handler and set formatter
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

    # Create logger and set handler
    logger = logging.getLogger("H2iVDI")
    logger.addHandler(handler)

    # Set level
    if debug_level > 0:
        logger.setLevel(logging.DEBUG)
    # elif level == "warning":
    #     logger.setLevel(logging.WARNING)
    # elif level == "error":
    #     logger.setLevel(logging.ERROR)
    # elif level == "critical":
    #     logger.setLevel(logging.CRITICAL)
    else:
        print("logger level set to INFO")
        logger.setLevel(logging.INFO)
    logger._debug_level = debug_level

    if debug_level > 1:
        logger.debugL2 = lambda x: logger.debug(x)
    else:
        logger.debugL2 = lambda x: None

    return logger