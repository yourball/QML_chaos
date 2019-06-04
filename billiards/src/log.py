import sys

NO = 1000
WARNING = 40
INFO = 30
DEBUG = 20
TRASH = 10


def log(threshold, loglevel, message):
    # if loglevel >= threshold:
    #     print(message, file=sys.stderr)
    print(message)


def trash(loglevel, message):
    message += ' - TRASH'
    log(loglevel, TRASH, message)


def debug(loglevel, message):
    message += ' - DEBUG'
    log(loglevel, DEBUG, message)


def info(loglevel, message):
    message += ' - INFO'
    log(loglevel, INFO, message)


def warning(loglevel, message):
    message += ' - WARNING'
    log(loglevel, WARNING, message)
