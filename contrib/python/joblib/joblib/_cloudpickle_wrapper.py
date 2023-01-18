"""
Small shim of loky's cloudpickle_wrapper to avoid failure when
multiprocessing is not available.
"""


from ._multiprocessing_helpers import mp


def my_wrap_non_picklable_objects(obj, keep_wrapper=True):
    return obj


if mp is None:
    wrap_non_picklable_objects = my_wrap_non_picklable_objects
else:
    from .externals.loky import wrap_non_picklable_objects # noqa
