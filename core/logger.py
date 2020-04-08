## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import colorama
import logging
import os
import re
import sys
from core import tools


def get_default_logging_format(colorize=False, brackets=False):
    style = colorama.Style.DIM if colorize else ''
    # color = colorama.Fore.CYAN if colorize else ''
    color = colorama.Fore.WHITE if colorize else ''
    reset = colorama.Style.RESET_ALL if colorize else ''
    if brackets:
        result = "{}{}[%(asctime)s]{} %(message)s".format(style, color, reset)
    else:
        result = "{}{}%(asctime)s{} %(message)s".format(style, color, reset)
    return result


def get_default_logging_datefmt():
    return "%Y-%m-%d %H:%M:%S"


def log_module_info(module):
    lines = module.__str__().split("\n")
    for line in lines:
        logging.info(line)


class LogbookFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super(LogbookFormatter, self).__init__(fmt=fmt, datefmt=datefmt)
        self._re = re.compile(r"\033\[[0-9]+m")

    def remove_colors_from_msg(self, msg):
        msg = re.sub(self._re, "", msg)
        return msg

    def format(self, record=None):
        record.msg = self.remove_colors_from_msg(record.msg)
        return super(LogbookFormatter, self).format(record)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super(ConsoleFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record=None):
        indent = sys.modules[__name__].global_indent
        record.msg = " " * indent + record.msg
        return super(ConsoleFormatter, self).format(record)


class SkipLogbookFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.LOGBOOK


def configure_logging(filename=None):
    # set global indent level
    sys.modules[__name__].global_indent = 0

    # add custom tqdm logger
    tools.addLoggingLevel("LOGBOOK", 1000)

    # create logger
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = get_default_logging_format(colorize=True, brackets=False)
    datefmt = get_default_logging_datefmt()
    formatter = ConsoleFormatter(fmt=fmt, datefmt=datefmt)
    console.setFormatter(formatter)

    # Skip logging.tqdm requests for console outputs
    skip_logbook_filter = SkipLogbookFilter()
    console.addFilter(skip_logbook_filter)

    # add console to root_logger
    root_logger.addHandler(console)

    # add logbook
    if filename is not None:
        # ensure dir
        d = os.path.dirname(filename)
        if not os.path.exists(d):
            os.makedirs(d)

        # --------------------------------------------------------------------------------------
        # Configure handler that removes color codes from logbook
        # --------------------------------------------------------------------------------------
        logbook = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
        logbook.setLevel(logging.INFO)
        fmt = get_default_logging_format(colorize=False, brackets=True)
        logbook_formatter = LogbookFormatter(fmt=fmt, datefmt=datefmt)
        logbook.setFormatter(logbook_formatter)
        root_logger.addHandler(logbook)


class LoggingBlock:
    def __init__(self, title, emph=False):
        self._emph = emph
        bright = colorama.Style.BRIGHT
        cyan = colorama.Fore.CYAN
        reset = colorama.Style.RESET_ALL
        if emph:
            logging.info("%s==>%s %s%s%s" % (cyan, reset, bright, title, reset))
        else:
            logging.info(title)

    def __enter__(self):
        sys.modules[__name__].global_indent += 2
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.modules[__name__].global_indent -= 2
