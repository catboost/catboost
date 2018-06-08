import random


def reservoir_sampling(data, nsamples):
    result = []
    for i, entry in enumerate(data):
        if i < nsamples:
            result.append(entry)
        else:
            j = random.randint(0, i)
            if j < nsamples:
                result[j] = entry
    return result
