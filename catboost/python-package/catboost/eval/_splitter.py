"""
class Splitter.

Convenient tool for creating and working with folds.
"""

import random

from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile


class _Splitter(object):
    """
     Splitter needs providing some parameters to create folds and some "reader",
     that can read source.
    """
    _REST_SIZE = 100000

    def __init__(self, line_reader, column_description, seed, min_folds_count):
        self._line_reader = line_reader
        self._line_groups_ids, self._groups_ids = self._read_groups_ids()
        # line_groups_ids -- group ids of each line
        # groups_ids -- set of all group ids in dataset
        self._folds_storage = set()
        # keeps it for removing at the end of work.
        self._column_description = column_description
        self._min_folds_count = min_folds_count
        self._random = random.Random(seed)

    # line_reader -- Reader for getting lines from file. It have to support iteration through lines.
    # column_description -- The description of the features of dataset.

    def _read_groups_ids(self):
        """Find all groups in dataset and which group each line belongs."""
        line_groups_ids = []
        groups_ids = set()

        lines = self._line_reader.lines_generator()
        # Need to return group id of line and line as string.
        for group_id, _ in lines:
            line_groups_ids.append(group_id)
            groups_ids.add(group_id)
        return line_groups_ids, groups_ids

    def _make_learn_folds(self, fold_size, left_folds):
        """Prepare test sets for folds only for one permutation"""
        count_groups = len(self._groups_ids)
        if count_groups // self._min_folds_count < fold_size:
            raise AttributeError('The size of fold is too big: count_groups: {}, fold_size: {}. Const: {}'.format(
                count_groups, fold_size, self._min_folds_count)
            )

        permutation = sorted(self._groups_ids)
        self._random.shuffle(permutation)

        result = []
        current_count_folds = min(count_groups // fold_size, left_folds)
        for i in range(current_count_folds):
            result.append(set(permutation[i * fold_size: (i + 1) * fold_size]))
        return result

    def _write_folds(self, fold_storages, num, offset):
        """Learn_set contains numbers of lines. The method itself store relevant lines from dataset to fold storage."""

        generator = self._line_reader.lines_generator()
        # Need to return group id of line and line as string.

        for fold_storage in fold_storages:
            fold_storage.open()

        try:
            rest_folds = []
            rest_fold_file = self.create_fold(None, 'offset{}_rest'.format(offset), num)
            rest_fold_file.open()
            num += 1
            rest_size = 0

            for num_line, (_, line) in enumerate(generator):
                group_id = self._line_groups_ids[num_line]
                is_written = False
                for fold_storage in fold_storages:
                    if fold_storage.contains_group_id(group_id):
                        fold_storage.add(line)
                        is_written = True
                if not is_written:
                    rest_fold_file.add(line)
                    rest_size += 1
                    if rest_size >= self._REST_SIZE:
                        rest_folds.append(rest_fold_file)
                        rest_fold_file.close()
                        rest_fold_file = self.create_fold(None, 'offset{}_rest'.format(offset), num)
                        rest_fold_file.open()
                        rest_size = 0
                        num += 1

            if rest_size > 0:
                rest_fold_file.close()
                rest_folds.append(rest_fold_file)
            elif rest_fold_file.is_opened():
                rest_fold_file.close()
        finally:
            for fold_storage in fold_storages:
                fold_storage.close()

        return rest_folds

    def create_fold_sets(self, fold_size, folds_count):
        """Create all folds for all permutations."""
        folds = []
        passed_folds_count = 0

        while passed_folds_count < folds_count:
            folds.append(self._make_learn_folds(fold_size, folds_count - passed_folds_count))
            current_learn_folds = folds[-1]
            passed_folds_count += len(current_learn_folds)
        return folds

    def fold_groups_files_generator(self, folds_groups, fold_offset):
        """Create folds storages for all folds in folds_groups. Generator."""

        fold_num = 0
        for fold_group in folds_groups:
            learn_folds = []
            skipped_folds = []
            for learn_set in fold_group:
                fold_num += 1
                if fold_offset < fold_num:
                    fold_file = self.create_fold(learn_set, 'fold', fold_num)
                    learn_folds.append(fold_file)
                elif fold_offset >= fold_num:
                    fold_file = self.create_fold(learn_set, 'offset{}_skipped'.format(fold_offset), fold_num)
                    skipped_folds.append(fold_file)

            rest_folds = self._write_folds(learn_folds + skipped_folds, fold_num, fold_offset)
            yield learn_folds, skipped_folds, rest_folds

    def create_fold(self, fold_set, name, id):
        file_name = self.create_name_from_id(name, id)
        fold_file = _FoldFile(fold_set,
                              file_name,
                              sep=self._line_reader.get_separator(),
                              column_description=self._column_description)
        self._folds_storage.add(fold_file)
        return fold_file

    def clean_folds(self):
        for file in self._folds_storage:
            file.delete()

    def clean(self):
        FoldStorage.remove_dir()

    @staticmethod
    def create_name_from_id(name, id, offset=None, max_count_digits=4):
        if offset is not None:
            name = '{name}{:0>{max_count_digits}}_offset{offset}'.format(
                id,
                name=name,
                max_count_digits=max_count_digits,
                offset=offset
            )
        else:
            name = '{name}{:0>{max_count_digits}}'.format(id, name=name, max_count_digits=max_count_digits)
        return name
