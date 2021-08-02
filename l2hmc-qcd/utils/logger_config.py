import sys
import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console as RichConsole
from rich.theme import Theme

import warnings

import logging.config
#  from utils.logger import Logger

REPO = 'fthmc'

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
LOGS_DIR = Path(BASE_DIR, 'l2hmclogs')
DATA_DIR = Path(BASE_DIR, 'data')
MODEL_DIR = Path(BASE_DIR, 'model')
STORES_DIR = Path(BASE_DIR, 'stores')

# Local stores
BLOB_STORE = Path(STORES_DIR, 'blob')
FEATURE_STORE = Path(STORES_DIR, 'feature')
MODEL_REGISTRY = Path(STORES_DIR, 'model')

# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
FEATURE_STORE.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)


# Logger
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
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}


def in_notebook():
    """Check if we're currently in a jupyter notebook."""
    try:
        # pylint:disable=import-outside-toplevel
        from IPython import get_ipython
        try:
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except AttributeError:
            return False
    except ImportError:
        return False
    return True


logging.config.dictConfig(logging_config)
logger = logging.getLogger('root')

theme = {}
if in_notebook():
    theme = {
        'repr.number': 'bold #87ff00',
        'repr.attrib_name': 'bold #ff5fff',
        'repr.str': 'italic #FFFF00',
    }


with_jupyter = in_notebook()
#  console = RichConsole(record=False, log_path=False,
#                        force_jupyter=with_jupyter,
#                        force_terminal=(not with_jupyter),
#                        log_time_format='[%x %X] ',
#                        theme=Theme(theme))#, width=width)
logger.handlers[0] = RichHandler(markup=True,
                                 #  console=console,
                                 show_path=False,
                                 rich_tracebacks=True)
warnings.filterwarnings('once', 'UserWarning')

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('seaborn').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('arviz').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
                                 #  console=Logger().console)

# Exclusion criteria
EXCLUDED_TAGS = []
