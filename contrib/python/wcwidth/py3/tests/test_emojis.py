# std imports
import os
import codecs

# 3rd party
import pytest

try:
    # python 2
    _ = unichr
except NameError:
    # python 3
    unichr = chr

# some tests cannot be done on some builds of python, where the internal
# unicode structure is limited to 0x10000 for memory conservation,
# "ValueError: unichr() arg not in range(0x10000) (narrow Python build)"
try:
    unichr(0x2fffe)
    NARROW_ONLY = False
except ValueError:
    NARROW_ONLY = True

# local
import wcwidth


def make_sequence_from_line(line):
    # convert '002A FE0F  ; ..' -> (0x2a, 0xfe0f) -> chr(0x2a) + chr(0xfe0f)
    return ''.join(unichr(int(cp, 16)) for cp in line.split(';', 1)[0].strip().split())


@pytest.mark.skipif(NARROW_ONLY, reason="Test cannot verify on python 'narrow' builds")
def emoji_zwj_sequence():
    u"""
    Emoji zwj sequence of four codepoints is just 2 cells.
    """
    phrase = (u"\U0001f469"   # Base, Category So, East Asian Width property 'W' -- WOMAN
              u"\U0001f3fb"   # Modifier, Category Sk, East Asian Width property 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-1-2
              u"\u200d"       # Joiner, Category Cf, East Asian Width property 'N'  -- ZERO WIDTH JOINER
              u"\U0001f4bb")  # Fused, Category So, East Asian Width peroperty 'W' -- PERSONAL COMPUTER
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    expect_length_each = (2, 0, 0, 2)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


@pytest.mark.skipif(NARROW_ONLY, reason="Test cannot verify on python 'narrow' builds")
def test_unfinished_zwj_sequence():
    u"""
    Ensure index-out-of-bounds does not occur for zero-width joiner without any following character
    """
    phrase = (u"\U0001f469"   # Base, Category So, East Asian Width property 'W' -- WOMAN
              u"\U0001f3fb"   # Modifier, Category Sk, East Asian Width property 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-1-2
              u"\u200d")      # Joiner, Category Cf, East Asian Width property 'N'  -- ZERO WIDTH JOINER
    expect_length_each = (2, 0, 0)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


@pytest.mark.skipif(NARROW_ONLY, reason="Test cannot verify on python 'narrow' builds")
def test_non_recommended_zwj_sequence():
    """
    Verify ZWJ is measured as though successful with characters that cannot be joined, wcwidth does not verify
    """
    phrase = (u"\U0001f469"   # Base, Category So, East Asian Width property 'W' -- WOMAN
              u"\U0001f3fb"   # Modifier, Category Sk, East Asian Width property 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-1-2
              u"\u200d")      # Joiner, Category Cf, East Asian Width property 'N'  -- ZERO WIDTH JOINER
    expect_length_each = (2, 0, 0)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


@pytest.mark.skipif(NARROW_ONLY, reason="Test cannot verify on python 'narrow' builds")
def test_another_emoji_zwj_sequence():
    phrase = (
        u"\u26F9"        # PERSON WITH BALL
        u"\U0001F3FB"    # EMOJI MODIFIER FITZPATRICK TYPE-1-2
        u"\u200D"        # ZERO WIDTH JOINER
        u"\u2640"        # FEMALE SIGN
        u"\uFE0F")       # VARIATION SELECTOR-16
    expect_length_each = (1, 0, 0, 1, 0)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


@pytest.mark.skipif(NARROW_ONLY, reason="Test cannot verify on python 'narrow' builds")
def test_longer_emoji_zwj_sequence():
    """
    A much longer emoji ZWJ sequence of 10 total codepoints is just 2 cells!

    Also test the same sequence in duplicate, verifying multiple VS-16 sequences
    in a single function call.
    """
    # 'Category Code', 'East Asian Width property' -- 'description'
    phrase = (u"\U0001F9D1"   # 'So', 'W' -- ADULT
              u"\U0001F3FB"   # 'Sk', 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-1-2
              u"\u200d"       # 'Cf', 'N' -- ZERO WIDTH JOINER
              u"\u2764"       # 'So', 'N' -- HEAVY BLACK HEART
              u"\uFE0F"       # 'Mn', 'A' -- VARIATION SELECTOR-16
              u"\u200d"       # 'Cf', 'N' -- ZERO WIDTH JOINER
              u"\U0001F48B"   # 'So', 'W' -- KISS MARK
              u"\u200d"       # 'Cf', 'N' -- ZERO WIDTH JOINER
              u"\U0001F9D1"   # 'So', 'W' -- ADULT
              u"\U0001F3FD"   # 'Sk', 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-4
    ) * 2
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    expect_length_each = (2, 0, 0, 1, 0, 0, 2, 0, 2, 0) * 2
    expect_length_phrase = 4

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def read_sequences_from_file(filename):
    fp = codecs.open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8')
    lines = [line.strip()
                for line in fp.readlines()
                if not line.startswith('#') and line.strip()]
    fp.close()
    sequences = [make_sequence_from_line(line) for line in lines]
    return lines, sequences


@pytest.mark.skipif(NARROW_ONLY, reason="Some sequences in text file are not compatible with 'narrow' builds")
def test_recommended_emoji_zwj_sequences():
    """
    Test wcswidth of all of the unicode.org-published emoji-zwj-sequences.txt
    """
    # given,
    lines, sequences = read_sequences_from_file('emoji-zwj-sequences.txt')

    errors = []
    # Exercise, track by zipping with original text file line, a debugging aide
    num = 0
    for sequence, line in zip(sequences, lines):
        num += 1
        measured_width = wcwidth.wcswidth(sequence)
        if measured_width != 2:
            errors.append({
                'expected_width': 2,
                'line': line,
                'measured_width': measured_width,
                'sequence': sequence,
            })

    # verify
    assert errors == []
    assert num >= 1468


def test_recommended_variation_16_sequences():
    """
    Test wcswidth of all of the unicode.org-published emoji-variation-sequences.txt
    """
    # given,
    lines, sequences = read_sequences_from_file('emoji-variation-sequences.txt')

    errors = []
    num = 0
    for sequence, line in zip(sequences, lines):
        num += 1
        if '\ufe0f' not in sequence:
            # filter for only \uFE0F (VS-16)
            continue
        measured_width = wcwidth.wcswidth(sequence)
        if measured_width != 2:
            errors.append({
                'expected_width': 2,
                'line': line,
                'measured_width': wcwidth.wcswidth(sequence),
                'sequence': sequence,
            })

    # verify
    assert errors == []
    assert num >= 742


def test_unicode_9_vs16():
    """Verify effect of VS-16 on unicode_version 9.0 and later"""
    phrase = (u"\u2640"        # FEMALE SIGN
              u"\uFE0F")       # VARIATION SELECTOR-16

    expect_length_each = (1, 0)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(wcwidth.wcwidth(w_char, unicode_version='9.0') for w_char in phrase)
    length_phrase = wcwidth.wcswidth(phrase, unicode_version='9.0')

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase

def test_unicode_8_vs16():
    """Verify that VS-16 has no effect on unicode_version 8.0 and earler"""
    phrase = (u"\u2640"        # FEMALE SIGN
              u"\uFE0F")       # VARIATION SELECTOR-16

    expect_length_each = (1, 0)
    expect_length_phrase = 1

    # exercise,
    length_each = tuple(wcwidth.wcwidth(w_char, unicode_version='8.0') for w_char in phrase)
    length_phrase = wcwidth.wcswidth(phrase, unicode_version='8.0')

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase