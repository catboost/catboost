"""
This module contains the abstractions for keeping small amount of data. It provides the interface for classes and a few
realisations.
"""

from __future__ import print_function

import os

from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from .utils import make_dirs_if_not_exists


# Interface
class FoldStorage(object):
    """
    Base class.
    """
    default_dir = 'folds'

    @staticmethod
    def remove_dir():
        """
        Remove default directory for folds if there're no files nut models. In other way it raises warning.

        Args:
            :return: Nothing.

        """
        try:
            if os.path.exists(_FoldFile.default_dir):
                os.rmdir(_FoldFile.default_dir)
        except OSError as err:
            get_eval_logger().warning(err.message)

    def __init__(self, fold, storage_name, sep, column_description):
        self._fold = fold
        self._storage_name = storage_name
        self._column_description = column_description
        self._sep = sep
        self._size = 0

    def get_separator(self):
        """
        Args:
            :return: (str) Delimiter for data used when we saved fold to file.

        """
        return self._sep

    def __str__(self):
        return self._storage_name

    def column_description(self):
        """
        Args:
            :return: (str) Path to the column description.

        """
        return self._column_description

    def contains_group_id(self, group_id):
        """
        Args:
            :param group_id: (int) The number of group we want to check.
            :return: True if fold contains line or lines with that group id.

        """
        return group_id in self._fold

    def open(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")

    def close(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")

    def delete(self):
        raise NotImplementedError("The base class don't have delete method. Please, use successor.")


class _FoldFile(FoldStorage):
    """
    FoldFile is the realisation of the interface of FoldStorage. It always saves data to file before reset them.
    All files place to the special directory 'folds'.
    """

    def __init__(self, fold, storage_name, sep, column_description):
        super(_FoldFile, self).__init__(
            fold, storage_name,
            sep=sep, column_description=column_description
        )
        self._file_path = os.path.join(self.default_dir,
                                       storage_name)
        self._prepare_path()
        self._lines = []
        self._file = None

    def _prepare_path(self):
        make_dirs_if_not_exists(self.default_dir)
        open(self._file_path, 'w').close()  # clean file

    def path(self):
        return self._file_path

    def add(self, line):
        self._size += 1
        print(line, file=self._file, end='')

    def add_all(self, lines):
        [self.add(line) for line in lines]

    def open(self):
        if self._file is None:
            self._file = open(self._file_path, mode='a')
        else:
            raise CatBoostError("File already opened {}".format(self._file_path))

    def is_opened(self):
        return self._file is not None

    def close(self):
        if self._file is None:
            raise CatBoostError("Trying to close None {}".format(self._file_path))

        self._file.close()
        self._file = None

    def delete(self):
        if self._file is not None:
            raise CatBoostError("Close file before delete")

        if os.path.exists(self._file_path):
            os.remove(self._file_path)
