# Sync content of this file with devtools/ya/core/error/__init__.py

TEMPORARY_ERROR_MESSAGES = [
    'Connection reset by peer',
    'Connection timed out',
    'Function not implemented',
    'I/O operation on closed file',
    'Internal Server Error',
    'Network connection closed unexpectedly',
    'Network is unreachable',
    'No route to host',
    'No space left on device',
    'Not enough space',
    'Temporary failure in name resolution',
    'The read operation timed out',
    'timeout: timed out',
]


# Node exit codes
class ExitCodes(object):
    TEST_FAILED = 10
    COMPILATION_FAILED = 11
    INFRASTRUCTURE_ERROR = 12
    NOT_RETRIABLE_ERROR = 13
    YT_STORE_FETCH_ERROR = 14


def merge_exit_codes(exit_codes):
    return max(e if e >= 0 else 1 for e in exit_codes) if exit_codes else 0


def is_temporary_error(exc):
    import logging
    logger = logging.getLogger(__name__)

    if getattr(exc, 'temporary', False):
        logger.debug("Exception has temporary attribute: %s", exc)
        return True

    import errno
    err = getattr(exc, 'errno', None)

    if err == errno.ECONNREFUSED or err == errno.ENETUNREACH:
        logger.debug("Exception has errno attribute: %s (errno=%s)", exc, err)
        return True

    import socket

    if isinstance(exc, socket.timeout) or isinstance(getattr(exc, 'reason', None), socket.timeout):
        logger.debug("Socket timeout exception: %s", exc)
        return True

    if isinstance(exc, socket.gaierror):
        logger.debug("Getaddrinfo exception: %s", exc)
        return True

    import urllib2

    if isinstance(exc, urllib2.HTTPError) and exc.code in (429, ):
        logger.debug("urllib2.HTTPError: %s", exc)
        return True

    import httplib

    if isinstance(exc, httplib.IncompleteRead):
        logger.debug("IncompleteRead exception: %s", exc)
        return True

    exc_str = str(exc)

    for message in TEMPORARY_ERROR_MESSAGES:
        if message in exc_str:
            logger.debug("Found temporary error pattern (%s): %s", message, exc_str)
            return True

    return False
