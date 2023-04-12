# coding=utf-8

import pytest
import six

from library.python import strings


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
    assert strings.to_basestring('str') == 'str'
    assert strings.to_basestring(u'юникод') == u'юникод'
    if six.PY2:  # __str__ should return str not bytes in Python3
        assert strings.to_basestring(Convertible()) == Convertible.text
        assert strings.to_basestring(ConvertibleToUnicodeOnly()) == Convertible.text
        assert strings.to_basestring(ConvertibleToStrOnly()) == Convertible.text_utf8
        assert strings.to_basestring(NonConvertible())


def test_to_unicode():
    assert strings.to_unicode(u'юникод') == u'юникод'
    assert strings.to_unicode('str') == u'str'
    assert strings.to_unicode(u'строка'.encode('utf-8')) == u'строка'
    assert strings.to_unicode(u'строка'.encode('cp1251'), 'cp1251') == u'строка'
    if six.PY2:  # __str__ should return str not bytes in Python3
        assert strings.to_unicode(Convertible()) == Convertible.text
        assert strings.to_unicode(ConvertibleToUnicodeOnly()) == Convertible.text
        with pytest.raises(UnicodeDecodeError):
            strings.to_unicode(ConvertibleToStrOnly())
        with pytest.raises(UnicodeDecodeError):
            strings.to_unicode(NonConvertible())


def test_to_unicode_errors_replace():
    assert strings.to_unicode(u'abcабв'.encode('utf-8'), 'ascii')
    assert strings.to_unicode(u'абв'.encode('utf-8'), 'ascii')


def test_to_str():
    assert strings.to_str('str') == 'str' if six.PY2 else b'str'
    assert strings.to_str(u'unicode') == 'unicode' if six.PY2 else b'unicode'
    assert strings.to_str(u'юникод') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод', 'cp1251') == u'юникод'.encode('cp1251')
    if six.PY2:
        assert strings.to_str(Convertible()) == Convertible.text_utf8
        with pytest.raises(UnicodeEncodeError):
            strings.to_str(ConvertibleToUnicodeOnly())
        assert strings.to_str(ConvertibleToStrOnly()) == Convertible.text_utf8
        with pytest.raises(UnicodeEncodeError):
            strings.to_str(NonConvertible())


def test_to_str_errors_replace():
    assert strings.to_str(u'abcабв', 'ascii')
    assert strings.to_str(u'абв', 'ascii')


def test_to_str_transcode():
    assert strings.to_str('str', from_enc='ascii') == 'str' if six.PY2 else b'str'
    assert strings.to_str('str', from_enc='utf-8') == 'str' if six.PY2 else b'str'

    assert strings.to_str(u'юникод'.encode('utf-8'), from_enc='utf-8') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('utf-8'), to_enc='utf-8', from_enc='utf-8') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('utf-8'), to_enc='cp1251', from_enc='utf-8') == u'юникод'.encode('cp1251')

    assert strings.to_str(u'юникод'.encode('cp1251'), from_enc='cp1251') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('cp1251'), to_enc='cp1251', from_enc='cp1251') == u'юникод'.encode('cp1251')
    assert strings.to_str(u'юникод'.encode('cp1251'), to_enc='utf-8', from_enc='cp1251') == u'юникод'.encode('utf-8')

    assert strings.to_str(u'юникод'.encode('koi8-r'), from_enc='koi8-r') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('koi8-r'), to_enc='koi8-r', from_enc='koi8-r') == u'юникод'.encode('koi8-r')
    assert strings.to_str(u'юникод'.encode('koi8-r'), to_enc='cp1251', from_enc='koi8-r') == u'юникод'.encode('cp1251')


def test_to_str_transcode_wrong():
    assert strings.to_str(u'юникод'.encode('utf-8'), from_enc='cp1251')
    assert strings.to_str(u'юникод'.encode('cp1251'), from_enc='utf-8')


def test_to_str_transcode_disabled():
    # No transcoding enabled, set from_enc to enable
    assert strings.to_str(u'юникод'.encode('utf-8'), to_enc='utf-8') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('utf-8'), to_enc='cp1251') == u'юникод'.encode('utf-8')
    assert strings.to_str(u'юникод'.encode('cp1251'), to_enc='utf-8') == u'юникод'.encode('cp1251')
    assert strings.to_str(u'юникод'.encode('cp1251'), to_enc='cp1251') == u'юникод'.encode('cp1251')
    assert strings.to_str(u'юникод'.encode('cp1251'), to_enc='koi8-r') == u'юникод'.encode('cp1251')
    assert strings.to_str(u'юникод'.encode('koi8-r'), to_enc='cp1251') == u'юникод'.encode('koi8-r')


def test_stringize_deep():
    assert strings.stringize_deep(
        {
            'key 1': 'value 1',
            u'ключ 2': u'значение 2',
            'list': [u'ключ 2', 'key 1', (u'к', 2)],
        }
    ) == {
        'key 1' if six.PY2 else b'key 1': 'value 1' if six.PY2 else b'value 1',
        u'ключ 2'.encode('utf-8'): u'значение 2'.encode('utf-8'),
        ('list' if six.PY2 else b'list'): [
            u'ключ 2'.encode('utf-8'),
            'key 1' if six.PY2 else b'key 1',
            (u'к'.encode('utf-8'), 2),
        ],
    }


def test_stringize_deep_doesnt_transcode():
    assert strings.stringize_deep(
        {
            u'ключ 1'.encode('utf-8'): u'значение 1'.encode('utf-8'),
            u'ключ 2'.encode('cp1251'): u'значение 2'.encode('cp1251'),
        }
    ) == {
        u'ключ 1'.encode('utf-8'): u'значение 1'.encode('utf-8'),
        u'ключ 2'.encode('cp1251'): u'значение 2'.encode('cp1251'),
    }


def test_stringize_deep_nested():
    assert strings.stringize_deep(
        {
            'key 1': 'value 1',
            u'ключ 2': {
                'subkey 1': 'value 1',
                u'подключ 2': u'value 2',
            },
        }
    ) == {
        'key 1' if six.PY2 else b'key 1': 'value 1' if six.PY2 else b'value 1',
        u'ключ 2'.encode('utf-8'): {
            ('subkey 1' if six.PY2 else b'subkey 1'): 'value 1' if six.PY2 else b'value 1',
            u'подключ 2'.encode('utf-8'): u'value 2'.encode('utf-8'),
        },
    }


def test_stringize_deep_plain():
    assert strings.stringize_deep('str') == 'str' if six.PY2 else b'str'
    assert strings.stringize_deep(u'юникод') == u'юникод'.encode('utf-8')
    assert strings.stringize_deep(u'юникод'.encode('utf-8')) == u'юникод'.encode('utf-8')


def test_stringize_deep_nonstr():
    with pytest.raises(TypeError):
        strings.stringize_deep(Convertible(), relaxed=False)
    x = Convertible()
    assert x == strings.stringize_deep(x)


def test_unicodize_deep():
    assert strings.unicodize_deep(
        {
            'key 1': 'value 1',
            u'ключ 2': u'значение 2',
            u'ключ 3'.encode('utf-8'): u'значение 3'.encode('utf-8'),
        }
    ) == {
        u'key 1': u'value 1',
        u'ключ 2': u'значение 2',
        u'ключ 3': u'значение 3',
    }


def test_unicodize_deep_nested():
    assert strings.unicodize_deep(
        {
            'key 1': 'value 1',
            u'ключ 2': {
                'subkey 1': 'value 1',
                u'подключ 2': u'значение 2',
                u'подключ 3'.encode('utf-8'): u'значение 3'.encode('utf-8'),
            },
        }
    ) == {
        u'key 1': u'value 1',
        u'ключ 2': {
            u'subkey 1': u'value 1',
            u'подключ 2': u'значение 2',
            u'подключ 3': u'значение 3',
        },
    }


def test_unicodize_deep_plain():
    assert strings.unicodize_deep('str') == u'str'
    assert strings.unicodize_deep(u'юникод') == u'юникод'
    assert strings.unicodize_deep(u'юникод'.encode('utf-8')) == u'юникод'


def test_unicodize_deep_nonstr():
    with pytest.raises(TypeError):
        strings.unicodize_deep(Convertible(), relaxed=False)
    x = Convertible()
    assert x == strings.stringize_deep(x)


truncate_utf_8_data = [
    ("hello", 5, None, None, "hello"),
    ("hello", 6, None, None, "hello"),
    ("hello", 4, None, None, "h..."),
    ("hello", 4, None, "", "hell"),
    ("hello", 4, None, ".", "hel."),
    ("hello", 4, strings.Whence.End, ".", "hel."),
    ("hello", 5, strings.Whence.Start, None, "hello"),
    ("hello", 4, strings.Whence.Start, None, "...o"),
    ("hello", 4, strings.Whence.Start, ".", ".llo"),
    ("yoloha", 5, strings.Whence.Middle, None, "y...a"),
    ("hello", 5, strings.Whence.Middle, None, "hello"),
    ("hello", 4, strings.Whence.Middle, None, "h..."),
    ("hello", 4, strings.Whence.Middle, ".", "he.o"),
    # destroyed symbol code must be removed
    ("меледа", 4, None, None, "..."),
    ("меледа", 5, None, None, "м..."),
    ("меледа", 7, None, None, "ме..."),
    ("меледа", 12, None, None, "меледа"),
    ("меледа", 4, None, ".", "м."),
    ("меледа", 5, None, "ак", "ак"),
    ("меледа", 6, None, "ак", "мак"),
    ("меледа", 4, strings.Whence.Start, None, "..."),
    ("меледа", 5, strings.Whence.Start, None, "...а"),
    ("меледа", 12, strings.Whence.Start, None, "меледа"),
    ("меледа", 9, strings.Whence.Start, ".", ".леда"),
    ("меледа", 10, strings.Whence.Start, ".", ".леда"),
    # half code from symbol 'м' plus half from symbol 'а' - nothing in the end
    ("меледа", 5, strings.Whence.Middle, None, "..."),
    ("меледа", 6, strings.Whence.Middle, None, "м..."),
    ("меледа", 7, strings.Whence.Middle, None, "м...а"),
    ("меледа", 12, strings.Whence.Middle, None, "меледа"),
    ("меледа", 8, strings.Whence.Middle, ".", "ме.а"),
    ("меледа", 9, strings.Whence.Middle, ".", "ме.да"),
    ("меледа", 10, strings.Whence.Middle, ".", "ме.да"),
    ("меледа", 11, strings.Whence.Middle, ".", "ме.да"),
    (u"меледа", 6, strings.Whence.Middle, None, "м..."),
    (u"меледа", 12, strings.Whence.Middle, None, "меледа"),
    (u"меледа", 8, strings.Whence.Middle, ".", "ме.а"),
]


@pytest.mark.parametrize("data, limit, Whence, msg, expected", truncate_utf_8_data)
def test_truncate_utf_8_text(data, limit, Whence, msg, expected):
    assert strings.truncate(data, limit, Whence, msg) == expected


def test_truncate_utf_8_text_wrong_limit():
    with pytest.raises(AssertionError):
        strings.truncate("hell", 2)

    with pytest.raises(AssertionError):
        strings.truncate("hello", 4, msg="long msg")
