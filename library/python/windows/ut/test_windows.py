# coding=utf-8

import errno
import os
import pytest
import six

import library.python.strings
import library.python.windows


def gen_error_access_denied():
    if library.python.windows.on_win():
        err = WindowsError()
        err.errno = errno.EACCES
        err.strerror = ''
        err.winerror = library.python.windows.ERRORS['ACCESS_DENIED']
    else:
        err = OSError()
        err.errno = errno.EACCES
        err.strerror = os.strerror(err.errno)
    err.filename = 'unknown/file'
    raise err


def test_errorfix_buggy():
    @library.python.windows.errorfix
    def erroneous_func():
        gen_error_access_denied()

    with pytest.raises(OSError) as errinfo:
        erroneous_func()
    assert errinfo.value.errno == errno.EACCES
    assert errinfo.value.filename == 'unknown/file'
    assert isinstance(errinfo.value.strerror, six.string_types)
    assert errinfo.value.strerror


def test_errorfix_explicit():
    @library.python.windows.errorfix
    def erroneous_func():
        if library.python.windows.on_win():
            err = WindowsError()
            err.winerror = library.python.windows.ERRORS['ACCESS_DENIED']
        else:
            err = OSError()
        err.errno = errno.EACCES
        err.strerror = 'Some error description'
        err.filename = 'unknown/file'
        raise err

    with pytest.raises(OSError) as errinfo:
        erroneous_func()
    assert errinfo.value.errno == errno.EACCES
    assert errinfo.value.filename == 'unknown/file'
    assert errinfo.value.strerror == 'Some error description'


def test_errorfix_decoding_cp1251():
    @library.python.windows.errorfix
    def erroneous_func():
        model_msg = u'Какое-то описание ошибки'
        if library.python.windows.on_win():
            err = WindowsError()
            err.strerror = library.python.strings.to_str(model_msg, 'cp1251')
        else:
            err = OSError()
            err.strerror = library.python.strings.to_str(model_msg)
        raise err

    with pytest.raises(OSError) as errinfo:
        erroneous_func()
    error_msg = errinfo.value.strerror
    if not isinstance(errinfo.value.strerror, six.text_type):
        error_msg = library.python.strings.to_unicode(error_msg)
    assert error_msg == u'Какое-то описание ошибки'


def test_diehard():
    @library.python.windows.diehard(library.python.windows.ERRORS['ACCESS_DENIED'], tries=5)
    def erroneous_func(errors):
        try:
            gen_error_access_denied()
        except Exception as e:
            errors.append(e)
            raise

    raised_errors = []
    with pytest.raises(OSError) as errinfo:
        erroneous_func(raised_errors)
    assert errinfo.value.errno == errno.EACCES
    assert any(e.errno == errno.EACCES for e in raised_errors)
    assert raised_errors and errinfo.value == raised_errors[-1]
    if library.python.windows.on_win():
        assert len(raised_errors) == 5
    else:
        assert len(raised_errors) == 1
