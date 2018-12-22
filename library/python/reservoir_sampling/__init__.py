import random


def reservoir_sampling(data, nsamples, prng=None):
    if prng is None:
        prng = random

    result = []
    for i, entry in enumerate(data):
        if i < nsamples:
            result.append(entry)
        else:
            j = prng.randint(0, i)
            if j < nsamples:
                result[j] = entry
    return result
