import json
import os
import random
import shutil

import numpy as np

__all__ = [
    'binary_path',
    'test_output_path',
    'remove_time_from_json',
    'DelayedTee',
    'permute_dataset_columns',
    'generate_random_labeled_set'
]

try:
    import yatest
    binary_path = yatest.common.binary_path
    test_output_path = yatest.common.test_output_path

except ImportError:
    def binary_path(*path):
        return os.path.join(os.environ["BINARY_PATH"], *path)

    def test_output_path(*path):
        return os.path.join(os.getcwd(), *path)


def remove_time_from_json(filename):
    with open(filename) as f:
        log = json.load(f)
    iterations = log['iterations']
    for i, iter_info in enumerate(iterations):
        for key in ['remaining_time', 'passed_time']:
            if key in iter_info.keys():
                del iter_info[key]
    with open(filename, 'w') as f:
        json.dump(log, f, sort_keys=True)
    return filename


# rewinds dst_stream to the start of the captured output so you can read it
class DelayedTee(object):

    def __init__(self, src_stream, dst_stream):
        self.src_stream = src_stream
        self.dst_stream = dst_stream

    def __enter__(self):
        self.src_stream.flush()
        self._old_src_stream = os.dup(self.src_stream.fileno())
        self._old_dst_stream_pos = self.dst_stream.tell()
        os.dup2(self.dst_stream.fileno(), self.src_stream.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        self.src_stream.flush()
        os.dup2(self._old_src_stream, self.src_stream.fileno())
        self.dst_stream.seek(self._old_dst_stream_pos)
        shutil.copyfileobj(self.dst_stream, self.src_stream)
        self.dst_stream.seek(self._old_dst_stream_pos)


def permute_dataset_columns(test_pool_path, cd_path, seed=123):
    permuted_test_path = test_output_path('permuted_test')
    permuted_cd_path = test_output_path('permuted_cd')
    generator = random.Random(seed)
    column_count = len(open(test_pool_path).readline().split('\t'))
    permutation = list(range(column_count))
    generator.shuffle(permutation)
    with open(cd_path) as original_cd, open(permuted_cd_path, 'w') as permuted_cd:
        for line in original_cd:
            line = line.strip()
            if not line:
                continue
            index, rest = line.split('\t', 1)
            permuted_cd.write('{}\t{}\n'.format(permutation.index(int(index)), rest))
    with open(test_pool_path) as test_pool, open(permuted_test_path, 'w') as permuted_test:
        for line in test_pool:
            splitted = line.strip().split('\t')
            permuted_test.write('\t'.join([splitted[i] for i in permutation]) + '\n')

    return permuted_test_path, permuted_cd_path


def generate_random_labeled_set(nrows, nvals, labels, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    label = prng.choice(labels, [nrows, 1])
    feature = prng.random_sample([nrows, nvals])
    return np.concatenate([label, feature], axis=1)

