# coding=utf-8

import pytest

import library.python.strings


class Convertible(object):
    text = u'текст'
    text_utf8 = text.encode('utf-8')

    def __unicode__(self):
        return self.text

    def __str__(self):
        return self.text_utf8


class ConvertibleToUnicodeOnly(Convertible):
    def __str__(self):
        return self.text.encode('ascii')


class ConvertibleToStrOnly(Convertible):
    def __unicode__(self):
        return self.text_utf8.decode('ascii')


class NonConvertible(ConvertibleToUnicodeOnly, ConvertibleToStrOnly):
    pass


def test_to_basestring():
    assert library.python.strings.to_basestring('str') == 'str'
    assert library.python.strings.to_basestring(u'юникод') == u'юникод'
    assert library.python.strings.to_basestring(Convertible()) == Convertible.text
    assert library.python.strings.to_basestring(ConvertibleToUnicodeOnly()) == Convertible.text
    assert library.python.strings.to_basestring(ConvertibleToStrOnly()) == Convertible.text_utf8
    assert library.python.strings.to_basestring(NonConvertible())


def test_to_unicode():
    assert library.python.strings.to_unicode(u'юникод') == u'юникод'
    assert library.python.strings.to_unicode('str') == u'str'
    assert library.python.strings.to_unicode(u'строка'.encode('utf-8')) == u'строка'
    assert library.python.strings.to_unicode(u'строка'.encode('cp1251'), 'cp1251') == u'строка'
    assert library.python.strings.to_unicode(Convertible()) == Convertible.text
    assert library.python.strings.to_unicode(ConvertibleToUnicodeOnly()) == Convertible.text
    with pytest.raises(UnicodeDecodeError):
        library.python.strings.to_unicode(ConvertibleToStrOnly())
    with pytest.raises(UnicodeDecodeError):
        library.python.strings.to_unicode(NonConvertible())


def test_to_unicode_errors_replace():
    assert library.python.strings.to_unicode(u'abcабв'.encode('utf-8'), 'ascii')
    assert library.python.strings.to_unicode(u'абв'.encode('utf-8'), 'ascii')


def test_to_str():
    assert library.python.strings.to_str('str') == 'str'
    assert library.python.strings.to_str(u'unicode') == 'unicode'
    assert library.python.strings.to_str(u'юникод') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод', 'cp1251') == u'юникод'.encode('cp1251')
    assert library.python.strings.to_str(Convertible()) == Convertible.text_utf8
    with pytest.raises(UnicodeEncodeError):
        library.python.strings.to_str(ConvertibleToUnicodeOnly())
    assert library.python.strings.to_str(ConvertibleToStrOnly()) == Convertible.text_utf8
    with pytest.raises(UnicodeEncodeError):
        library.python.strings.to_str(NonConvertible())


def test_to_str_errors_replace():
    assert library.python.strings.to_str(u'abcабв', 'ascii')
    assert library.python.strings.to_str(u'абв', 'ascii')


def test_to_str_transcode():
    assert library.python.strings.to_str('str', from_enc='ascii') == 'str'
    assert library.python.strings.to_str('str', from_enc='utf-8') == 'str'

    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), from_enc='utf-8') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), to_enc='utf-8', from_enc='utf-8') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), to_enc='cp1251', from_enc='utf-8') == u'юникод'.encode('cp1251')

    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), from_enc='cp1251') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), to_enc='cp1251', from_enc='cp1251') == u'юникод'.encode('cp1251')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), to_enc='utf-8', from_enc='cp1251') == u'юникод'.encode('utf-8')

    assert library.python.strings.to_str(u'юникод'.encode('koi8-r'), from_enc='koi8-r') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('koi8-r'), to_enc='koi8-r', from_enc='koi8-r') == u'юникод'.encode('koi8-r')
    assert library.python.strings.to_str(u'юникод'.encode('koi8-r'), to_enc='cp1251', from_enc='koi8-r') == u'юникод'.encode('cp1251')


def test_to_str_transcode_wrong():
    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), from_enc='cp1251')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), from_enc='utf-8')


def test_to_str_transcode_disabled():
    # No transcoding enabled, set from_enc to enable
    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), to_enc='utf-8') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('utf-8'), to_enc='cp1251') == u'юникод'.encode('utf-8')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), to_enc='utf-8') == u'юникод'.encode('cp1251')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), to_enc='cp1251') == u'юникод'.encode('cp1251')
    assert library.python.strings.to_str(u'юникод'.encode('cp1251'), to_enc='koi8-r') == u'юникод'.encode('cp1251')
    assert library.python.strings.to_str(u'юникод'.encode('koi8-r'), to_enc='cp1251') == u'юникод'.encode('koi8-r')


def test_stringize_deep():
    assert library.python.strings.stringize_deep({
        'key 1': 'value 1',
        u'ключ 2': u'значение 2',
        'list': [u'ключ 2', 'key 1', (u'к', 2)]
    }) == {
        'key 1': 'value 1',
        u'ключ 2'.encode('utf-8'): u'значение 2'.encode('utf-8'),
        'list': [u'ключ 2'.encode('utf-8'), 'key 1', (u'к'.encode('utf-8'), 2)]
    }


def test_stringize_deep_doesnt_transcode():
    assert library.python.strings.stringize_deep({
        u'ключ 1'.encode('utf-8'): u'значение 1'.encode('utf-8'),
        u'ключ 2'.encode('cp1251'): u'значение 2'.encode('cp1251'),
    }) == {
        u'ключ 1'.encode('utf-8'): u'значение 1'.encode('utf-8'),
        u'ключ 2'.encode('cp1251'): u'значение 2'.encode('cp1251'),
    }


def test_stringize_deep_nested():
    assert library.python.strings.stringize_deep({
        'key 1': 'value 1',
        u'ключ 2': {
            'subkey 1': 'value 1',
            u'подключ 2': u'value 2',
        },
    }) == {
        'key 1': 'value 1',
        u'ключ 2'.encode('utf-8'): {
            'subkey 1': 'value 1',
            u'подключ 2'.encode('utf-8'): u'value 2'.encode('utf-8'),
        },
    }


def test_stringize_deep_plain():
    assert library.python.strings.stringize_deep('str') == 'str'
    assert library.python.strings.stringize_deep(u'юникод') == u'юникод'.encode('utf-8')
    assert library.python.strings.stringize_deep(u'юникод'.encode('utf-8')) == u'юникод'.encode('utf-8')


def test_stringize_deep_nonstr():
    with pytest.raises(TypeError):
        library.python.strings.stringize_deep(Convertible(), relaxed=False)
    x = Convertible()
    assert x == library.python.strings.stringize_deep(x)


def test_unicodize_deep():
    assert library.python.strings.unicodize_deep({
        'key 1': 'value 1',
        u'ключ 2': u'значение 2',
        u'ключ 3'.encode('utf-8'): u'значение 3'.encode('utf-8'),
    }) == {
        u'key 1': u'value 1',
        u'ключ 2': u'значение 2',
        u'ключ 3': u'значение 3',
    }


def test_unicodize_deep_nested():
    assert library.python.strings.unicodize_deep({
        'key 1': 'value 1',
        u'ключ 2': {
            'subkey 1': 'value 1',
            u'подключ 2': u'значение 2',
            u'подключ 3'.encode('utf-8'): u'значение 3'.encode('utf-8'),
        },
    }) == {
        u'key 1': u'value 1',
        u'ключ 2': {
            u'subkey 1': u'value 1',
            u'подключ 2': u'значение 2',
            u'подключ 3': u'значение 3',
        },
    }


def test_unicodize_deep_plain():
    assert library.python.strings.unicodize_deep('str') == u'str'
    assert library.python.strings.unicodize_deep(u'юникод') == u'юникод'
    assert library.python.strings.unicodize_deep(u'юникод'.encode('utf-8')) == u'юникод'


def test_unicodize_deep_nonstr():
    with pytest.raises(TypeError):
        library.python.strings.unicodize_deep(Convertible(), relaxed=False)
    x = Convertible()
    assert x == library.python.strings.stringize_deep(x)
