from six.moves import cPickle
import copy


def deepcopy(obj):
    try:
        return cPickle.loads(cPickle.dumps(obj, -1))

    except cPickle.PicklingError:
        return copy.deepcopy(obj)
