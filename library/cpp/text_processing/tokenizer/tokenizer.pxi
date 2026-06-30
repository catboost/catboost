# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from six import string_types

cimport cython  # noqa
from cython.operator cimport dereference

from libcpp cimport bool as bool_t

from util.generic.string cimport TString, TStringBuf
from util.generic.ptr cimport THolder, MakeHolder
from util.generic.vector cimport TVector
from util.string.cast cimport FromString, ToString

from library.cpp.text_processing.tokenizer.tokenizer cimport (
    ETokenType, ESeparatorType, ESubTokensPolicy, ETokenProcessPolicy, ELanguage, TTokenizer, TTokenizerOptions)


cdef extern from "library/cpp/langs/langs.h" nogil:
    cdef ELanguage LanguageByNameOrDie(TStringBuf language) except +


cdef TTokenizerOptions CreateTokenizerOptions(
    lowercasing,
    lemmatizing,
    number_process_policy,
    number_token,
    separator_type,
    delimiter,
    split_by_set,
    skip_empty,
    token_types,
    sub_tokens_policy,
    languages,
) except *:
    cdef TTokenizerOptions tokenizer_options

    if lowercasing is not None:
        tokenizer_options.Lowercasing = <bool_t>lowercasing
    if lemmatizing is not None:
        tokenizer_options.Lemmatizing = <bool_t>lemmatizing
    if number_process_policy is not None:
        tokenizer_options.NumberProcessPolicy = FromString[ETokenProcessPolicy](to_arcadia_string(number_process_policy))
    if number_token is not None:
        tokenizer_options.NumberToken = to_arcadia_string(number_token)
    if separator_type is not None:
        tokenizer_options.SeparatorType = FromString[ESeparatorType](to_arcadia_string(separator_type))
    if delimiter is not None:
        tokenizer_options.Delimiter = to_arcadia_string(delimiter)
    if split_by_set is not None:
        tokenizer_options.SplitBySet = <bool_t>split_by_set
    if skip_empty is not None:
        tokenizer_options.SkipEmpty = <bool_t>skip_empty
    if token_types is not None:
        assert isinstance(token_types, list)
        tokenizer_options.TokenTypes.clear()
        for tok_type_str in token_types:
            tokenizer_options.TokenTypes.insert(
                FromString[ETokenType](
                    to_arcadia_string(tok_type_str)
                ))
    if sub_tokens_policy is not None:
        tokenizer_options.SubTokensPolicy = FromString[ESubTokensPolicy](to_arcadia_string(sub_tokens_policy))
    if languages is not None:
        if not isinstance(languages, list):
            languages = [languages]
        tokenizer_options.Languages.clear()
        for language in languages:
            tokenizer_options.Languages.push_back(LanguageByNameOrDie(to_arcadia_string(language)))

    return tokenizer_options


@cython.embedsignature(True)
cdef class Tokenizer:
    cdef THolder[TTokenizer] __tokenizer

    def __dealloc__(self):
        self.__tokenizer.Reset()

    def __init__(
        self,
        lowercasing=None,
        lemmatizing=None,
        number_process_policy=None,
        number_token=None,
        separator_type=None,
        delimiter=None,
        split_by_set=None,
        skip_empty=None,
        token_types=None,
        sub_tokens_policy=None,
        languages=None,
    ):
        """
        Tokenizer.

        Parameters
        ----------
        lowercasing : bool, optional (default=None)
            Convert tokens to lower case.

        lemmatizing : bool, optional (default=None)
            Perform lemmatization on tokens.

        number_process_policy : string, optional (default=None)
            The strategy to process numeric tokens. Possible values:
                - 'Skip' - Skip all numeric tokens.
                - 'LeaveAsIs' - Leave all numeric tokens as is.
                - 'Replace' - Replace all numeric tokens with a single special token.
                    This token you can specify with the `number_token` parameter.

        number_token : string, optional (default=None)
            The special token that is used to replace all numeric tokens with.
            This option can be used if the selected numeric tokens processing strategy is 'Replace'.

        separator_type : string, optional (default=None)
            The tokenization method. Possible values:
                - 'ByDelimiter' - Split by delimiter.
                - 'BySense' - Split the string by sense.

        delimiter : string, optional (default=None)
            The symbol that is considered to be the delimiter.
            Should be used if the separator_type parameter is set to BySense.

        split_by_set : bool, optional (default=None)
            Use each single character in the `delimiter` option as an individual delimiter.
            This parameter allows to use multiple delimiters.

        skip_empty : bool, optional (default=None)
            Skip all empty tokens.

        token_types : list of strings, optional (default=None)
            The types of tokens that should be kept after the tokenization.
            Should be used if the separator_type parameter is set to BySense.
            Possible values: 'Word', 'Digit', 'Punctuation','SentenceBreak', 'ParagraphBreak', 'Unknown'.

        sub_tokens_policy : string, optional (default=None)
            The subtokens processing policy.
            Should be used if the `separator_type` parameter is set to 'BySense'.
            Possible values:
                - 'SingleToken' - All subtokens are interpreted as a single token.
                - 'SeveralTokens' - All subtokens are interpreted as several tokens.

        languages : list of strings, optional (default=None)
            The list of languages to use.
            Should be used if the `separator_type` parameter is set to 'BySense'.
            Leave the value of this parameter empty to use all available languages (significantly slows down the procedure).
        """

        cdef TTokenizerOptions tokenizer_options = CreateTokenizerOptions(
            lowercasing,
            lemmatizing,
            number_process_policy,
            number_token,
            separator_type,
            delimiter,
            split_by_set,
            skip_empty,
            token_types,
            sub_tokens_policy,
            languages,
        )

        self.__tokenizer = MakeHolder[TTokenizer](tokenizer_options)

    def tokenize(self, string, types=False):
        """
        Tokenize input string.

        Parameters
        ----------
        string : string
            Input string.

        types : bool, optional (default=None)
            Need to return token types also.

        Returns
        ----------
        tokens: list of strings or list of tuples (token, token_type) if `types` == True.
        """
        if not string:
            return []
        assert isinstance(string, string_types)
        cdef TString arc_string = to_arcadia_string(string)
        cdef TVector[TString] tokens
        cdef TVector[ETokenType] tokenTypes
        dereference(self.__tokenizer.Get()).Tokenize(arc_string, &tokens, &tokenTypes)

        def get_result(i):
            if types:
                return to_str(tokens[i]), to_str(ToString(tokenTypes[i]))
            else:
                return to_str(tokens[i])

        return [get_result(i) for i in xrange(tokens.size())]
