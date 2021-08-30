from __future__ import unicode_literals
import os
import time
import subprocess
import warnings
import tempfile
import pickle

import pytest


class PicklableMixin(object):
    def _get_nobj_bytes(self, obj, dump_kwargs, load_kwargs):
        """
        Pickle and unpickle an object using ``pickle.dumps`` / ``pickle.loads``
        """
        pkl = pickle.dumps(obj, **dump_kwargs)
        return pickle.loads(pkl, **load_kwargs)

    def _get_nobj_file(self, obj, dump_kwargs, load_kwargs):
        """
        Pickle and unpickle an object using ``pickle.dump`` / ``pickle.load`` on
        a temporary file.
        """
        with tempfile.TemporaryFile('w+b') as pkl:
            pickle.dump(obj, pkl, **dump_kwargs)
            pkl.seek(0)         # Reset the file to the beginning to read it
            nobj = pickle.load(pkl, **load_kwargs)

        return nobj

    def assertPicklable(self, obj, singleton=False, asfile=False,
                        dump_kwargs=None, load_kwargs=None):
        """
        Assert that an object can be pickled and unpickled. This assertion
        assumes that the desired behavior is that the unpickled object compares
        equal to the original object, but is not the same object.
        """
        get_nobj = self._get_nobj_file if asfile else self._get_nobj_bytes
        dump_kwargs = dump_kwargs or {}
        load_kwargs = load_kwargs or {}

        nobj = get_nobj(obj, dump_kwargs, load_kwargs)
        if not singleton:
            self.assertIsNot(obj, nobj)
        self.assertEqual(obj, nobj)


class TZContextBase(object):
    """
    Base class for a context manager which allows changing of time zones.

    Subclasses may define a guard variable to either block or or allow time
    zone changes by redefining ``_guard_var_name`` and ``_guard_allows_change``.
    The default is that the guard variable must be affirmatively set.

    Subclasses must define ``get_current_tz`` and ``set_current_tz``.
    """
    _guard_var_name = "DATEUTIL_MAY_CHANGE_TZ"
    _guard_allows_change = True

    def __init__(self, tzval):
        self.tzval = tzval
        self._old_tz = None

    @classmethod
    def tz_change_allowed(cls):
        """
        Class method used to query whether or not this class allows time zone
        changes.
        """
        guard = bool(os.environ.get(cls._guard_var_name, False))

        # _guard_allows_change gives the "default" behavior - if True, the
        # guard is overcoming a block. If false, the guard is causing a block.
        # Whether tz_change is allowed is therefore the XNOR of the two.
        return guard == cls._guard_allows_change

    @classmethod
    def tz_change_disallowed_message(cls):
        """ Generate instructions on how to allow tz changes """
        msg = ('Changing time zone not allowed. Set {envar} to {gval} '
               'if you would like to allow this behavior')

        return msg.format(envar=cls._guard_var_name,
                          gval=cls._guard_allows_change)

    def __enter__(self):
        if not self.tz_change_allowed():
            msg = self.tz_change_disallowed_message()
            pytest.skip(msg)

            # If this is used outside of a test suite, we still want an error.
            raise ValueError(msg)  # pragma: no cover

        self._old_tz = self.get_current_tz()
        self.set_current_tz(self.tzval)

    def __exit__(self, type, value, traceback):
        if self._old_tz is not None:
            self.set_current_tz(self._old_tz)

        self._old_tz = None

    def get_current_tz(self):
        raise NotImplementedError

    def set_current_tz(self):
        raise NotImplementedError


class TZEnvContext(TZContextBase):
    """
    Context manager that temporarily sets the `TZ` variable (for use on
    *nix-like systems). Because the effect is local to the shell anyway, this
    will apply *unless* a guard is set.

    If you do not want the TZ environment variable set, you may set the
    ``DATEUTIL_MAY_NOT_CHANGE_TZ_VAR`` variable to a truthy value.
    """
    _guard_var_name = "DATEUTIL_MAY_NOT_CHANGE_TZ_VAR"
    _guard_allows_change = False

    def get_current_tz(self):
        return os.environ.get('TZ', UnsetTz)

    def set_current_tz(self, tzval):
        if tzval is UnsetTz and 'TZ' in os.environ:
            del os.environ['TZ']
        else:
            os.environ['TZ'] = tzval

        time.tzset()


class TZWinContext(TZContextBase):
    """
    Context manager for changing local time zone on Windows.

    Because the effect of this is system-wide and global, it may have
    unintended side effect. Set the ``DATEUTIL_MAY_CHANGE_TZ`` environment
    variable to a truthy value before using this context manager.
    """
    def get_current_tz(self):
        p = subprocess.Popen(['tzutil', '/g'], stdout=subprocess.PIPE)

        ctzname, err = p.communicate()
        ctzname = ctzname.decode()     # Popen returns

        if p.returncode:
            raise OSError('Failed to get current time zone: ' + err)

        return ctzname

    def set_current_tz(self, tzname):
        p = subprocess.Popen('tzutil /s "' + tzname + '"')

        out, err = p.communicate()

        if p.returncode:
            raise OSError('Failed to set current time zone: ' +
                          (err or 'Unknown error.'))


###
# Utility classes
class NotAValueClass(object):
    """
    A class analogous to NaN that has operations defined for any type.
    """
    def _op(self, other):
        return self             # Operation with NotAValue returns NotAValue

    def _cmp(self, other):
        return False

    __add__ = __radd__ = _op
    __sub__ = __rsub__ = _op
    __mul__ = __rmul__ = _op
    __div__ = __rdiv__ = _op
    __truediv__ = __rtruediv__ = _op
    __floordiv__ = __rfloordiv__ = _op

    __lt__ = __rlt__ = _op
    __gt__ = __rgt__ = _op
    __eq__ = __req__ = _op
    __le__ = __rle__ = _op
    __ge__ = __rge__ = _op


NotAValue = NotAValueClass()


class ComparesEqualClass(object):
    """
    A class that is always equal to whatever you compare it to.
    """

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    __req__ = __eq__
    __rne__ = __ne__
    __rle__ = __le__
    __rge__ = __ge__
    __rlt__ = __lt__
    __rgt__ = __gt__


ComparesEqual = ComparesEqualClass()


class UnsetTzClass(object):
    """ Sentinel class for unset time zone variable """
    pass


UnsetTz = UnsetTzClass()
