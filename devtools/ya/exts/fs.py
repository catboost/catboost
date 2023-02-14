import logging
import os
import stat

from library.python.fs import *  # noqa

if supports_clone():  # noqa F405
    from library.python.fs.clonefile import macos_clone_file  # noqa

logger = logging.getLogger(__name__)


def remove_tree_with_perm_update(dir):
    def remove(safe):
        try:
            if safe:
                remove_tree_safe(dir)  # noqa
            else:
                ensure_removed(dir)  # noqa
            return True
        except OSError as e:
            logger.debug('Error while removing dir: %s', e)

        return False

    if remove(safe=False):
        return

    logger.debug('Trying to change permissions while removing dir: %s', dir)

    try:
        os.chmod(dir, os.stat(dir).st_mode | stat.S_IWRITE)
        for root, dirs, files in os.walk(dir):
            for file_name in files:
                extracted_path = os.path.join(root, file_name)
                # For Windows read-only files.
                os.chmod(extracted_path, os.stat(extracted_path).st_mode | stat.S_IWRITE)
            for dir_name in dirs:
                extracted_dir = os.path.join(root, dir_name)
                os.chmod(extracted_dir, os.stat(extracted_dir).st_mode | stat.S_IWRITE)
    except Exception as e:
        logger.debug('Error while changing permissions: %s', e)

    remove(safe=True)
