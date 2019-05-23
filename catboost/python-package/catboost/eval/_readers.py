"""
Simple file reader. It intends to read lines from big files. Also it provides the group id of each line it reads.
"""

import pandas as pd
from .utils import series_to_line


class _BaseReader(object):
    def __init__(self, sep, group_feature_num):
        self._sep = sep
        self._group_feature_num = group_feature_num

    def get_separator(self):
        return self._sep

    def lines_generator(self):
        raise NotImplementedError("The base class don't have any lines_reader")

    def pack_lines_generator(self, pack_size):
        lines = []
        group_ids = []
        current_pack_size = 0

        lines_generator = self.lines_generator()
        for group_id, line in lines_generator:
            group_ids.append(group_id)
            lines.append(line)
            current_pack_size += 1
            if current_pack_size == pack_size:
                yield group_ids, lines
                lines = []
                group_ids = []
                current_pack_size = 0
        if current_pack_size != 0:
            yield group_ids, lines


class _SimpleStreamingFileReader(_BaseReader):
    def __init__(self, file_name, sep, has_header, group_feature_num=None):
        super(_SimpleStreamingFileReader, self).__init__(sep, group_feature_num)
        self._has_header = has_header
        self._file_name = file_name

    def lines_generator(self):
        with open(self._file_name, 'r') as file:
            if self._has_header:
                file.readline()
            for num, line in enumerate(file):
                if self._group_feature_num is None:
                    group_id = num
                else:
                    features = line.strip().split(self._sep, self._group_feature_num + 1)
                    group_id = features[self._group_feature_num]
                yield int(float(group_id)), line


# Can't handle big data. Can be used for tests.
class _SimpleDataReader(_BaseReader):
    def __init__(self, data, sep, group_feature_num=None):
        super(_SimpleDataReader, self).__init__(sep, group_feature_num)
        self._data = pd.DataFrame(data)

    def lines_generator(self):
        for num, (index, line) in enumerate(self._data.iterrows()):
            if self._group_feature_num is None:
                yield num, series_to_line(line, self._sep) + '\n'
            else:
                yield line.iloc[self._group_feature_num], series_to_line(line, self._sep) + '\n'

    def get_matrix(self):
        return self._data
