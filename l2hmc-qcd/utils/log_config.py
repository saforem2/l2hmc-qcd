import os
import sys
import logging
from pathlib import Path
# Logger
BASE_DIR = Path(str(os.getcwd()))
logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "rich": {
            "class": "rich.logging.RichHandler",
        },
        "ipython": {
            "class": "rich.logging.RichHandler",
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(os.getcwd()).joinpath("info.log")),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
            "mode": "a",
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(os.getcwd()).joinpath("error.log")),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
            "mode": "a",
        },
    },
    "loggers": {
        "l2hmc": {
            "handlers": ["rich", "info", "error"],
            "level": logging.DEBUG,
            # "propagate": True,
        },
        "jupyter": {
            "handlers": ["ipython"],
            "level": logging.DEBUG,
        },
    },
}
