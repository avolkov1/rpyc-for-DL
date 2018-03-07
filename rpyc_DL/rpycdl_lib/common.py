'''
'''
import os
import errno
import sys
import logging


__all__ = ('get_logger', 'mkdir_p')


def get_logger(name, level=None):
    level = level or logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def mkdir_p(path):
    """ 'mkdir -p' in Python """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
