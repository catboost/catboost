"""
Run chardet on a bunch of documents and see that we get the correct encodings.

:author: Dan Blanchard
:author: Ian Cordasco
"""


import textwrap
from difflib import ndiff
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
from pprint import pformat
from unicodedata import normalize

try:
    import hypothesis.strategies as st
    from hypothesis import Verbosity, assume, given, settings

    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False
import pytest  # pylint: disable=import-error

import chardet
from chardet.metadata.languages import LANGUAGES

# TODO: Restore Hungarian encodings (iso-8859-2 and windows-1250) after we
#       retrain model.
MISSING_ENCODINGS = {
    "iso-8859-2",
    "iso-8859-6",
    "windows-1250",
    "windows-1254",
    "windows-1256",
}
EXPECTED_FAILURES = {
    "tests/iso-8859-9-turkish/_ude_1.txt",
    "tests/iso-8859-9-turkish/_ude_2.txt",
    "tests/iso-8859-9-turkish/divxplanet.com.xml",
    "tests/iso-8859-9-turkish/subtitle.srt",
    "tests/iso-8859-9-turkish/wikitop_tr_ISO-8859-9.txt",
}


def gen_test_params():
    """Yields tuples of paths and encodings to use for test_encoding_detection"""
    import yatest.common
    base_path = yatest.common.work_path('test_data/tests')
    for encoding in listdir(base_path):
        path = join(base_path, encoding)
        # Skip files in tests directory
        if not isdir(path):
            continue
        # Remove language suffixes from encoding if present
        encoding = encoding.lower()
        for language in sorted(LANGUAGES.keys()):
            postfix = "-" + language.lower()
            if encoding.endswith(postfix):
                encoding = encoding.rpartition(postfix)[0]
                break
        # Skip directories for encodings we don't handle yet.
        if encoding in MISSING_ENCODINGS:
            continue
        # Test encoding detection for each file we have of encoding for
        for file_name in listdir(path):
            ext = splitext(file_name)[1].lower()
            if ext not in [".html", ".txt", ".xml", ".srt"]:
                continue
            full_path = join(path, file_name)
            test_case = full_path, encoding
            name_test = full_path.split("/test_data/")[-1]
            if name_test in EXPECTED_FAILURES:
                test_case = pytest.param(*test_case, marks=pytest.mark.xfail, id=name_test)
            else:
                test_case = pytest.param(*test_case, id=name_test)
            yield test_case


@pytest.mark.parametrize("file_name, encoding", gen_test_params())
def test_encoding_detection(file_name, encoding):
    with open(file_name, "rb") as f:
        input_bytes = f.read()
        result = chardet.detect(input_bytes)
        try:
            expected_unicode = input_bytes.decode(encoding)
        except LookupError:
            expected_unicode = ""
        try:
            detected_unicode = input_bytes.decode(result["encoding"])
        except (LookupError, UnicodeDecodeError, TypeError):
            detected_unicode = ""
    if result:
        encoding_match = (result["encoding"] or "").lower() == encoding
    else:
        encoding_match = False
    # Only care about mismatches that would actually result in different
    # behavior when decoding
    expected_unicode = normalize("NFKC", expected_unicode)
    detected_unicode = normalize("NFKC", detected_unicode)
    if not encoding_match and expected_unicode != detected_unicode:
        wrapped_expected = "\n".join(textwrap.wrap(expected_unicode, 100)) + "\n"
        wrapped_detected = "\n".join(textwrap.wrap(detected_unicode, 100)) + "\n"
        diff = "".join(
            [
                line
                for line in ndiff(
                    wrapped_expected.splitlines(True), wrapped_detected.splitlines(True)
                )
                if not line.startswith(" ")
            ][:20]
        )
        all_encodings = chardet.detect_all(input_bytes, ignore_threshold=True)
    else:
        diff = ""
        encoding_match = True
        all_encodings = [result]
    assert encoding_match, (
        f"Expected {encoding}, but got {result} for {file_name}.  First 20 "
        f"lines with character differences: \n{diff}\n"
        f"All encodings: {pformat(all_encodings)}"
    )


@pytest.mark.parametrize("file_name, encoding", gen_test_params())
def test_encoding_detection_rename_legacy(file_name, encoding):
    with open(file_name, "rb") as f:
        input_bytes = f.read()
        result = chardet.detect(input_bytes, should_rename_legacy=True)
        try:
            expected_unicode = input_bytes.decode(encoding)
        except LookupError:
            expected_unicode = ""
        try:
            detected_unicode = input_bytes.decode(result["encoding"])
        except (LookupError, UnicodeDecodeError, TypeError):
            detected_unicode = ""
    if result:
        encoding_match = (result["encoding"] or "").lower() == encoding
    else:
        encoding_match = False
    # Only care about mismatches that would actually result in different
    # behavior when decoding
    expected_unicode = normalize("NFKD", expected_unicode)
    detected_unicode = normalize("NFKD", detected_unicode)
    if not encoding_match and expected_unicode != detected_unicode:
        wrapped_expected = "\n".join(textwrap.wrap(expected_unicode, 100)) + "\n"
        wrapped_detected = "\n".join(textwrap.wrap(detected_unicode, 100)) + "\n"
        diff = "".join(
            [
                line
                for line in ndiff(
                    wrapped_expected.splitlines(True), wrapped_detected.splitlines(True)
                )
                if not line.startswith(" ")
            ][:20]
        )
        all_encodings = chardet.detect_all(
            input_bytes, ignore_threshold=True, should_rename_legacy=True
        )
    else:
        diff = ""
        encoding_match = True
        all_encodings = [result]
    assert encoding_match, (
        f"Expected {encoding}, but got {result} for {file_name}.  First 20 "
        f"lines of character differences: \n{diff}\n"
        f"All encodings: {pformat(all_encodings)}"
    )


if HAVE_HYPOTHESIS:

    class JustALengthIssue(Exception):
        pass

    @pytest.mark.xfail
    @given(
        st.text(min_size=1),
        st.sampled_from(
            [
                "ascii",
                "utf-8",
                "utf-16",
                "utf-32",
                "iso-8859-7",
                "iso-8859-8",
                "windows-1255",
            ]
        ),
        st.randoms(),
    )
    @settings(max_examples=200)
    def test_never_fails_to_detect_if_there_is_a_valid_encoding(txt, enc, rnd):
        try:
            data = txt.encode(enc)
        except UnicodeEncodeError:
            assume(False)
        detected = chardet.detect(data)["encoding"]
        if detected is None:
            with pytest.raises(JustALengthIssue):

                @given(st.text(), random=rnd)
                @settings(verbosity=Verbosity.quiet, max_examples=50)
                def string_poisons_following_text(suffix):
                    try:
                        extended = (txt + suffix).encode(enc)
                    except UnicodeEncodeError:
                        assume(False)
                    result = chardet.detect(extended)
                    if result and result["encoding"] is not None:
                        raise JustALengthIssue()

    @given(
        st.text(min_size=1),
        st.sampled_from(
            [
                "ascii",
                "utf-8",
                "utf-16",
                "utf-32",
                "iso-8859-7",
                "iso-8859-8",
                "windows-1255",
            ]
        ),
        st.randoms(),
    )
    @settings(max_examples=200)
    def test_detect_all_and_detect_one_should_agree(txt, enc, _):
        try:
            data = txt.encode(enc)
        except UnicodeEncodeError:
            assume(False)
        try:
            result = chardet.detect(data)
            results = chardet.detect_all(data)
            assert result["encoding"] == results[0]["encoding"]
        except Exception as exc:
            raise RuntimeError(f"{result} != {results}") from exc
