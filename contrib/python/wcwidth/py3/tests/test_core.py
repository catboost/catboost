# coding: utf-8
"""Core tests for wcwidth module. isort:skip_file"""
try:
    # std import
    import importlib.metadata as importmeta
except ImportError:
    # 3rd party for python3.7 and earlier
    import importlib_metadata as importmeta

# local
import wcwidth

# 3rd party
import pytest

# some tests cannot be done on some builds of python, where the internal
# unicode structure is limited to 0x10000 for memory conservation,
# "ValueError: unichr() arg not in range(0x10000) (narrow Python build)"
try:
    # python 2
    _ = unichr
except NameError:
    # python 3
    unichr = chr
try:
    unichr(0x2fffe)
    NARROW_ONLY = False
except ValueError:
    NARROW_ONLY = True


def test_package_version():
    """wcwidth.__version__ is expected value."""
    # given,
    expected = importmeta.version('wcwidth')

    # exercise,
    result = wcwidth.__version__

    # verify.
    assert result == expected


def test_empty_string():
    """
    Test empty string is OK.

    https://github.com/jquast/wcwidth/issues/24
    """
    phrase = ""
    expect_length_each = 0
    expect_length_phrase = 0

    # exercise,
    length_each = wcwidth.wcwidth(phrase)
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def basic_string_type():
    """
    This is a python 2-specific test of the basic "string type"

    Such strings cannot contain anything but ascii in python2.
    """
    # given,
    phrase = 'hello\x00world'
    expect_length_each = (1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
    expect_length_phrase = sum(expect_length_each)

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_hello_jp():
    u"""
    Width of Japanese phrase: コンニチハ, セカイ!

    Given a phrase of 5 and 3 Katakana ideographs, joined with
    3 English-ASCII punctuation characters, totaling 11, this
    phrase consumes 19 cells of a terminal emulator.
    """
    # given,
    phrase = u'コンニチハ, セカイ!'
    expect_length_each = (2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1)
    expect_length_phrase = sum(expect_length_each)

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_wcswidth_substr():
    """
    Test wcswidth() optional 2nd parameter, ``n``.

    ``n`` determines at which position of the string
    to stop counting length.
    """
    # given,
    phrase = u'コンニチハ, セカイ!'
    end = 7
    expect_length_each = (2, 2, 2, 2, 2, 1, 1,)
    expect_length_phrase = sum(expect_length_each)

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))[:end]
    length_phrase = wcwidth.wcswidth(phrase, end)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_null_width_0():
    """NULL (0) reports width 0."""
    # given,
    phrase = u'abc\x00def'
    expect_length_each = (1, 1, 1, 0, 1, 1, 1)
    expect_length_phrase = sum(expect_length_each)

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase, len(phrase))

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_control_c0_width_negative_1():
    """How the API reacts to CSI (Control sequence initiate).

    An example of bad fortune, this terminal sequence is a width of 0
    on all terminals, but wcwidth doesn't parse Control-Sequence-Inducer
    (CSI) sequences.

    Also the "legacy" posix functions wcwidth and wcswidth return -1 for
    any string containing the C1 control character \x1b (ESC).
    """
    # given,
    phrase = u'\x1b[0m'
    expect_length_each = (-1, 1, 1, 1)
    expect_length_phrase = -1

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify, though this is actually *0* width for a terminal emulator
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_combining_width():
    """Simple test combining reports total width of 4."""
    # given,
    phrase = u'--\u05bf--'
    expect_length_each = (1, 1, 0, 1, 1)
    expect_length_phrase = 4

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_combining_cafe():
    u"""Phrase cafe + COMBINING ACUTE ACCENT is café of length 4."""
    phrase = u"cafe\u0301"
    expect_length_each = (1, 1, 1, 1, 0)
    expect_length_phrase = 4

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_combining_enclosing():
    u"""CYRILLIC CAPITAL LETTER A + COMBINING CYRILLIC HUNDRED THOUSANDS SIGN is of length 1."""
    phrase = u"\u0410\u0488"
    expect_length_each = (1, 0)
    expect_length_phrase = 1

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_balinese_script():
    u"""
    Balinese kapal (ship) is length 3.

    This may be an example that is not yet correctly rendered by any terminal so
    far, like devanagari.
    """
    phrase = (u"\u1B13"    # Category 'Lo', EAW 'N' -- BALINESE LETTER KA
              u"\u1B28"    # Category 'Lo', EAW 'N' -- BALINESE LETTER PA KAPAL
              u"\u1B2E"    # Category 'Lo', EAW 'N' -- BALINESE LETTER LA
              u"\u1B44")   # Category 'Mc', EAW 'N' -- BALINESE ADEG ADEG
    expect_length_each = (1, 1, 1, 0)
    expect_length_phrase = 3

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_kr_jamo_filler():
    u"""
    Jamo filler is 0 width.

    According to https://www.unicode.org/L2/L2006/06310-hangul-decompose9.pdf this character and others
    like it, ``\uffa0``, ``\u1160``, ``\u115f``, ``\u1160``, are not commonly viewed with a terminal,
    seems it doesn't matter whether it is implemented or not, they are not typically used !
    """
    phrase = u"\u1100\u1160"
    expect_length_each = (2, 1)
    expect_length_phrase = 3

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


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
def test_longer_emoji_zwj_sequence():
    """
    A much longer emoji ZWJ sequence of 10 total codepoints is just 2 cells!
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
              u"\U0001F3FD")  # 'Sk', 'W' -- EMOJI MODIFIER FITZPATRICK TYPE-4

    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    expect_length_each = (2, 0, 0, 1, 0, 0, 2, 0, 2, 0)
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_devanagari_script():
    """
    Attempt to test the measurement width of Devanagari script.

    I believe this 'phrase' should be length 3.

    This is a difficult problem, and this library does not yet get it right,
    because we interpret the unicode data files programmatically, but they do
    not correctly describe how their terminal width is measured.

    There are very few Terminals that do!

    As of 2023,

    - iTerm2: correct length but individual characters are out of order and
              horizaontally misplaced as to be unreadable in its language when
              using 'Noto Sans' font.
    - mlterm: mixed results, it offers several options in the configuration
              dialog, "Xft", "Cario", and "Variable Column Width" have some
              effect, but with neither 'Noto Sans' or 'unifont', it is not
              recognizable as the Devanagari script it is meant to display.

    Previous testing with Devanagari documented at address https://benizi.com/vim/devanagari/

    See also, https://askubuntu.com/questions/8437/is-there-a-good-mono-spaced-font-for-devanagari-script-in-the-terminal
    """
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    # please note that document correctly points out that the final width cannot be determined
    # as a sum of each individual width, as this library currently performs with exception of
    # ZWJ, but I think it incorrectly gestures what a stateless call to wcwidth.wcwidth of
    # each codepoint *should* return.
    phrase = (u"\u0915"    # Akhand, Category 'Lo', East Asian Width property 'N' -- DEVANAGARI LETTER KA
              u"\u094D"    # Joiner, Category 'Mn', East Asian Width property 'N' -- DEVANAGARI SIGN VIRAMA
              u"\u0937"    # Fused, Category 'Lo', East Asian Width property 'N' -- DEVANAGARI LETTER SSA
              u"\u093F")   # MatraL, Category 'Mc', East Asian Width property 'N' -- DEVANAGARI VOWEL SIGN I
    # 23107-terminal-suppt.pdf suggests wcwidth.wcwidth should return (2, 0, 0, 1)
    expect_length_each = (1, 0, 1, 0)
    # I believe the final width *should* be 3.
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_tamil_script():
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    phrase = (u"\u0b95"    # Akhand, Category 'Lo', East Asian Width property 'N' -- TAMIL LETTER KA
              u"\u0bcd"    # Joiner, Category 'Mn', East Asian Width property 'N' -- TAMIL SIGN VIRAMA
              u"\u0bb7"    # Fused, Category 'Lo', East Asian Width property 'N' -- TAMIL LETTER SSA
              u"\u0bcc")   # MatraLR, Category 'Mc', East Asian Width property 'N' -- TAMIL VOWEL SIGN AU
    # 23107-terminal-suppt.pdf suggests wcwidth.wcwidth should return (3, 0, 0, 4)
    expect_length_each = (1, 0, 1, 0)

    # I believe the final width should be about 5 or 6.
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_kannada_script():
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    # |ರ್ಝೈ|
    # |123|
    phrase = (u"\u0cb0"    # Repha, Category 'Lo', East Asian Width property 'N' -- KANNADA LETTER RA
              u"\u0ccd"    # Joiner, Category 'Mn', East Asian Width property 'N' -- KANNADA SIGN VIRAMA
              u"\u0c9d"    # Base, Category 'Lo', East Asian Width property 'N' -- KANNADA LETTER JHA
              u"\u0cc8")   # MatraUR, Category 'Mc', East Asian Width property 'N' -- KANNADA VOWEL SIGN AI
    # 23107-terminal-suppt.pdf suggests should be (2, 0, 3, 1)
    expect_length_each = (1, 0, 1, 0)
    # I believe the correct final width *should* be 3 or 4.
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase


def test_kannada_script_2():
    # This test adapted from https://www.unicode.org/L2/L2023/23107-terminal-suppt.pdf
    # |ರ಼್ಚ|
    # |12|
    phrase = (u"\u0cb0"    # Base, Category 'Lo', East Asian Width property 'N' -- KANNADA LETTER RA
              u"\u0cbc"    # Nukta, Category 'Mn', East Asian Width property 'N' -- KANNADA SIGN NUKTA
              u"\u0ccd"    # Joiner, Category 'Lo', East Asian Width property 'N' -- KANNADA SIGN VIRAMA
              u"\u0c9a")   # Subjoin, Category 'Mc', East Asian Width property 'N' -- KANNADA LETTER CA
    # 23107-terminal-suppt.pdf suggests wcwidth.wcwidth should return (2, 0, 0, 1)
    expect_length_each = (1, 0, 0, 1)
    # I believe the final width is correct, but maybe for the wrong reasons!
    expect_length_phrase = 2

    # exercise,
    length_each = tuple(map(wcwidth.wcwidth, phrase))
    length_phrase = wcwidth.wcswidth(phrase)

    # verify.
    assert length_each == expect_length_each
    assert length_phrase == expect_length_phrase
