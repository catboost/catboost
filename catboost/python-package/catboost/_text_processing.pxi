# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from six import string_types

cimport cython  # noqa
from cython.operator cimport dereference

from libcpp cimport bool as bool_t
from libcpp cimport nullptr
from util.system.types cimport ui32

from util.generic.array_ref cimport TConstArrayRef
from util.generic.string cimport TString
from util.generic.ptr cimport THolder, MakeHolder, TIntrusivePtr
from util.generic.vector cimport TVector
from util.string.cast cimport FromString

from library.cpp.text_processing.tokenizer.tokenizer cimport TTokenizer, TTokenizerOptions, ETokenType
from library.cpp.text_processing.dictionary.dictionary cimport (
    IDictionary, TDictionary, TBpeDictionary, TTokenId, TFileInput, IInputStream, TFileOutput,
    ETokenLevelType, EEndOfWordTokenPolicy, EEndOfSentenceTokenPolicy, EUnknownTokenPolicy,
    TDictionaryOptions, TDictionaryBuilderOptions, TBpeDictionaryOptions, TDictionaryBuilder,
    EDictionaryType, EDictionaryType_FrequencyBased, EDictionaryType_Bpe)

import numpy as np

try:
    from pandas import Series
except ImportError:
    class Series(object):
        pass

include "library/cpp/text_processing/tokenizer/tokenizer.pxi"


ctypedef TBpeDictionary * TBpeDictionaryPtr


cdef extern from "library/cpp/text_processing/app_helpers/app_helpers.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef TIntrusivePtr[TDictionary] BuildDictionary(
        const TString& inputFilePath,
        const TDictionaryBuilderOptions& dictionaryBuilderOptions,
        const TDictionaryOptions& dictionaryOptions,
        const TTokenizerOptions& tokenizerOptions,
        bool_t useTokenizer,
        bool_t verbose
    ) except +

    cdef TIntrusivePtr[TBpeDictionary] BuildBpe(
        const TString& inputFilePath,
        const TDictionaryBuilderOptions& dictionaryBuilderOptions,
        const TDictionaryOptions& dictionaryOptions,
        const TBpeDictionaryOptions& bpeOptions,
        const TTokenizerOptions& tokenizerOptions,
        bool_t useTokenizer,
        bool_t verbose
    ) except +


cdef extern from * nogil:
    cdef T dynamic_cast[T](void *) except +


cdef TDictionaryBuilderOptions CreateDictionaryBuilderOptions(occurence_lower_bound, max_dictionary_size) except *:
    cdef TDictionaryBuilderOptions dict_builder_options

    if occurence_lower_bound is not None:
        dict_builder_options.OccurrenceLowerBound = occurence_lower_bound
    if max_dictionary_size is not None:
        dict_builder_options.MaxDictionarySize = max_dictionary_size

    return dict_builder_options


cdef TDictionaryOptions CreateDictionaryOptions(
    token_level_type,
    gram_order,
    skip_step,
    start_token_id,
    end_of_word_policy,
    end_of_sentence_policy,
) except *:
    cdef TDictionaryOptions dict_options

    if token_level_type is not None:
        dict_options.TokenLevelType = FromString[ETokenLevelType](to_arcadia_string(token_level_type))
    if gram_order is not None:
        dict_options.GramOrder = gram_order
    if skip_step is not None:
        dict_options.SkipStep = skip_step
    if start_token_id is not None:
        dict_options.StartTokenId = <TTokenId>start_token_id
    if end_of_word_policy is not None:
        dict_options.EndOfWordTokenPolicy = FromString[EEndOfWordTokenPolicy](to_arcadia_string(end_of_word_policy))
    if end_of_sentence_policy is not None:
        dict_options.EndOfSentenceTokenPolicy = FromString[EEndOfSentenceTokenPolicy](to_arcadia_string(end_of_sentence_policy))

    return dict_options


cdef TBpeDictionaryOptions CreateBpeDictionaryOptions(num_bpe_units, skip_unknown) except *:
    cdef TBpeDictionaryOptions bpe_dict_options

    if num_bpe_units is not None:
        bpe_dict_options.NumUnits = num_bpe_units
    if skip_unknown is not None:
        bpe_dict_options.SkipUnknown = skip_unknown

    return bpe_dict_options


def _ensure(condition, message=None):
    if not condition:
        raise Exception(message or 'condition does not hold')


@cython.embedsignature(True)
cdef class Dictionary:
    cdef THolder[IDictionary] __dictionary_holder
    cdef TDictionaryBuilderOptions __dict_builder_options
    cdef TDictionaryOptions __dict_options
    cdef TBpeDictionaryOptions __bpe_dict_options
    cdef EDictionaryType __dictionary_type

    def __check_dictionary_initialized(self):
        assert <bool_t>self.__dictionary_holder.Get(), "Dictionary should be initialized"

    def __dealloc__(self):
        self.__dictionary_holder.Reset()

    def __init__(
        self,
        token_level_type=None,
        gram_order=None,
        skip_step=None,
        start_token_id=None,
        end_of_word_policy=None,
        end_of_sentence_policy=None,
        occurence_lower_bound=None,
        max_dictionary_size=None,
        num_bpe_units=None,
        skip_unknown=None,
        dictionary_type='FrequencyBased',
    ):
        '''
        Dictionary.

        Parameters
        ----------
        token_level_type : string, optional (default=None)
            The token level type. This parameter defines what should be considered a separate token.
            Possible values: 'Word', 'Letter'.
        gram_order : int, optional (default=None)
            The number of words or letters in each token.
            For example, let's assume that it is required to build a dictionary
            for the following sentence: 'maybe some other time'. If the token level type is set to Word
            and this parameter is set to 2, the following tokens are formed: 'maybe some', 'some other', 'other time'.
        skip_step : int, optional (default=None)
            The number of words or letters to skip when joining them to tokens. This parameter takes effect if
            the value of the GramOrder parameter is strictly greater than 1. For example, let's assume that
            it is required to build a dictionary for the following sentence: 'maybe some other time'.
            If the token level type is set to Word, GramOrder is set to 2 and this parameter is set to 1,
            the following tokens are formed: 'maybe other', 'some time'.
        start_token_id : int, optional (default=None)
            The initial shift for the token identifier. For example, let's assume that
            it is required to build a dictionary for the following sentence: 'maybe some other time'.
            If this parameter is set to 42, the following identifiers are assigned to tokens:
            42 - 'maybe', 43 - 'some', 44 - 'other', 45 - 'time'.
        end_of_word_policy : string, optional (default=None)
            The policy for processing implicit tokens that point to the end of the word.
            Possible values: 'Skip', 'Insert'.
        end_of_sentence_policy : string, optional (default=None)
            The policy for processing implicit tokens that point to the end of the sentence.
            Possible values: 'Skip', 'Insert'.
        occurence_lower_bound : int, optional (default=None)
            The lower limit of token occurrences in the text to include it in the dictionary.
        max_dictionary_size : int, optional (default=None)
            The maximum dictionary size.
        num_bpe_units : int, optional (default=None)
            The number of token pairs that should be combined to a single token.
        skip_unknown : bool, optional (default=None)
            Skip unknown tokens when building the dictionary.
        dictionary_type : string, optional (default='FrequencyBased')
            Dictionary type. Possible values: 'FrequencyBased', 'Bpe'.
        '''

        self.__dict_options = CreateDictionaryOptions(token_level_type, gram_order, skip_step, start_token_id, end_of_word_policy, end_of_sentence_policy)
        self.__dict_builder_options = CreateDictionaryBuilderOptions(occurence_lower_bound, max_dictionary_size)
        self.__bpe_dict_options = CreateBpeDictionaryOptions(num_bpe_units, skip_unknown)
        self.__dictionary_type = FromString[EDictionaryType](to_arcadia_string(dictionary_type))

    cdef __fit_fb_from_array(self, data, const TTokenizerOptions& tokenizerOptions, bool_t useTokenizer):
        cdef THolder[TDictionaryBuilder] dictionaryBuilder = MakeHolder[TDictionaryBuilder](self.__dict_builder_options, self.__dict_options)
        cdef THolder[TTokenizer] tokenizer = MakeHolder[TTokenizer](tokenizerOptions)
        cdef TVector[TString] tokens
        msg = 'Expected string types, but got: {}'
        for line in data:
            tokens.clear()
            if isinstance(line, string_types):
                if useTokenizer:
                    dereference(tokenizer.Get()).Tokenize(to_arcadia_string(line), &tokens, <TVector[ETokenType]*>nullptr)
                else:
                    tokens.push_back(to_arcadia_string(line))
            elif isinstance(line, (list, np.ndarray, Series)):
                [_ensure(isinstance(token, string_types), msg.format(type(token))) for token in line]
                for token in line:
                    tokens.push_back(to_arcadia_string(token))
            dereference(dictionaryBuilder.Get()).Add(<TConstArrayRef[TString]>tokens);
        self.__dictionary_holder = THolder[IDictionary](dereference(dictionaryBuilder.Get()).FinishBuilding().Release())

    cdef __fit_fb(self, data, const TTokenizerOptions& tokenizerOptions, bool_t useTokenizer, bool_t verbose):
        if isinstance(data, string_types):
            self.__dictionary_holder = THolder[IDictionary](BuildDictionary(
                to_arcadia_string(data),
                self.__dict_builder_options,
                self.__dict_options,
                tokenizerOptions,
                useTokenizer,
                verbose
            ).Release())
        elif isinstance(data, (list, np.ndarray, Series)):
            self.__fit_fb_from_array(data, tokenizerOptions, useTokenizer)
        else:
            raise Exception('Unsupported data format.')


    cdef __fit_bpe(self, data, const TTokenizerOptions& tokenizerOptions, bool_t useTokenizer, bool_t verbose):
        if isinstance(data, string_types):
            self.__dictionary_holder = THolder[IDictionary](BuildBpe(
                to_arcadia_string(data),
                self.__dict_builder_options,
                self.__dict_options,
                self.__bpe_dict_options,
                tokenizerOptions,
                useTokenizer,
                verbose
            ).Release())
        else:
            raise Exception('Now you can fit dictionary from file.')

    cpdef fit(self, data, Tokenizer tokenizer=None, verbose=False):
        """
        Train dictionary.

        Parameters
        ----------
        data : string or list or numpy.ndarray or pandas.Series
            Input data.
            If string, giving the path to the file with text data.
            If list or numpy.ndarrays or pandas.Series, giving 1 or 2 dimensional array like text data.

        tokenizer : Tokenizer, optional (default=None)
            Tokenizer for text processing. If you specify it and pass 1-dimensional data,
            each element will be consider as sentence and will tokenize.

        verbose : bool, optional (default=None)
            Logging level.

        Returns
        ----------
        dictionary : Dictionary.
        """

        cdef TTokenizer * tokenizerImpl
        cdef TTokenizerOptions tokenizerOptions
        cdef bool_t useTokenizer = False
        if tokenizer is not None:
            useTokenizer = True
            tokenizerImpl = tokenizer.__tokenizer.Get()
            tokenizerOptions = dereference(tokenizerImpl).GetOptions()

        if self.__dictionary_type == EDictionaryType_FrequencyBased:
            self.__fit_fb(data, tokenizerOptions, useTokenizer, verbose)
        else:
            assert self.__dictionary_type == EDictionaryType_Bpe
            self.__fit_bpe(data, tokenizerOptions, useTokenizer, verbose)
        return self

    cpdef apply(self, data, Tokenizer tokenizer=None, unknown_token_policy=None):
        """
        Apply dictionary to text.

        Parameters
        ----------
        data : string or list or numpy.ndarray or pandas.Series
            Input data. Giving 0,1 or 2 dimensional array like text data.

        tokenizer : Tokenizer, optional (default=None)
            Tokenizer for text processing. If you specify it and pass 1-dimensional data,
            each element will be consider as sentence and will tokenize.

        unknown_token_policy : string, optional (default=None)
            Unknown token policy. Possible values:
                - 'Skip' - All unknown tokens will be skipped from result token_ids.
                - 'Insert' - Result token_ids array contains token_id of unknown tokens.

        Returns
        ----------
        token_ids : 1 or 2 dimensional array with token_ids.
        """

        self.__check_dictionary_initialized()

        need_to_extract = False
        if isinstance(data, string_types):
            data = [data]
            need_to_extract = True

        unknown_token_policy = 'Skip' if unknown_token_policy is None else unknown_token_policy
        cdef EUnknownTokenPolicy unknownTokenPolicy = FromString[EUnknownTokenPolicy](to_arcadia_string(unknown_token_policy))

        token_ids = []
        cdef TVector[TString] tokens
        cdef TVector[TTokenId] tokenIds
        for line in data:
            tokens.clear()
            tokenIds.clear()
            if isinstance(line, string_types):
                if tokenizer is not None:
                    line = tokenizer.tokenize(line)
                else:
                    line = [line]
            for token in line:
                tokens.push_back(to_arcadia_string(token))
            dereference(self.__dictionary_holder.Get()).Apply(TConstArrayRef[TString](tokens), &tokenIds, unknownTokenPolicy)
            token_ids.append([<int>tokenId for tokenId in tokenIds])

        if need_to_extract:
            token_ids = token_ids[0]

        return token_ids

    @property
    def size(self):
        """
        Dictionary size.

        Returns
        ----------
        size : int.
        """

        self.__check_dictionary_initialized()
        return dereference(self.__dictionary_holder.Get()).Size()

    def get_token(self, token_id):
        """
        Get token according on token_id.

        Parameters
        ----------
        token_id : int
            Token id.

        Returns
        ----------
        token : string.
        """

        cdef ui32 _token_id = token_id
        self.__check_dictionary_initialized()
        return to_str(dereference(self.__dictionary_holder.Get()).GetToken(_token_id))

    def get_tokens(self, token_ids):
        """
        Get tokens according on token_id.

        Parameters
        ----------
        token_ids : list
            Token ids

        Returns
        ----------
        tokens : list.
        """
        self.__check_dictionary_initialized()

        cdef TVector[TTokenId] tokenIds
        tokenIds.reserve(len(token_ids))
        for token_id in token_ids:
            tokenIds.push_back(token_id)

        cdef TVector[TString] tokens
        dereference(self.__dictionary_holder.Get()).GetTokens(<TConstArrayRef[TTokenId]>tokenIds, &tokens)
        return [to_str(token) for token in tokens]

    def get_top_tokens(self, top_size=None):
        """
        Get the top of most popular tokens.

        Parameters
        ----------
        top_size : int
            Top size.

        Returns
        ----------
        tokens : list.
        """
        self.__check_dictionary_initialized()
        cdef TVector[TString] top
        if top_size is None:
            top = dereference(self.__dictionary_holder.Get()).GetTopTokens()
        else:
            top = dereference(self.__dictionary_holder.Get()).GetTopTokens(top_size)
        return [to_str(s) for s in top]

    @property
    def unknown_token_id(self):
        """
        Get the identifier of the token, which is assigned to all words that are not found in the dictionary.

        Returns
        ----------
        size : int.
        """

        self.__check_dictionary_initialized()
        return <int>dereference(self.__dictionary_holder.Get()).GetUnknownTokenId()

    @property
    def end_of_sentence_token_id(self):
        """
        Get the identifier of the last token in the sentence.

        Returns
        ----------
        size : int.
        """

        self.__check_dictionary_initialized()
        return <int>dereference(self.__dictionary_holder.Get()).GetEndOfSentenceTokenId()

    @property
    def min_unused_token_id(self):
        """
        Get the smallest unused token identifier.

        Identifiers are assigned consistently to all input tokens. Some additional identifiers are
        reserved for internal needs. This method returns the first unused identifier.
        All further identifiers are assumed to be unassigned to any token.

        Returns
        ----------
        size : int.
        """

        self.__check_dictionary_initialized()
        return <int>dereference(self.__dictionary_holder.Get()).GetMinUnusedTokenId()

    def load(self, frequency_dict_path, bpe_path=None):
        """
        Load dictionary from file.

        Parameters
        ----------
        frequency_dict_path : string
            Frequency based dictionary file path.

        bpe_path : string, optional (default=None)
            Bpe dictionary file path.

        Returns
        ----------
        dictionary : Dictionary.
        """

        dictionary_type = 'FrequencyBased' if bpe_path is None else 'Bpe'
        self.__dictionary_type = FromString[EDictionaryType](to_arcadia_string(dictionary_type))

        cdef THolder[TFileInput] dict_file = MakeHolder[TFileInput](to_arcadia_string(frequency_dict_path))
        cdef THolder[TBpeDictionary] bpe_dictoinary = MakeHolder[TBpeDictionary]()

        if dictionary_type == 'FrequencyBased':
            self.__dictionary_holder.Reset(IDictionary.Load(<IInputStream*>dict_file.Get()).Release())
        else:
            assert dictionary_type == 'Bpe'
            dereference(bpe_dictoinary.Get()).Load(to_arcadia_string(frequency_dict_path), to_arcadia_string(bpe_path))
            self.__dictionary_holder.Reset(bpe_dictoinary.Release())
        return self


    def save(self, frequency_dict_path, bpe_path=None):
        """
        Save dictionary to file.

        Parameters
        ----------
        frequency_dict_path : string
            Frequency based dictionary file path.

        bpe_path : string, optional (default=None)
            Bpe dictionary file path.

        Returns
        ----------
        dictionary : Dictionary.
        """

        self.__check_dictionary_initialized()
        cdef THolder[TFileOutput] file = MakeHolder[TFileOutput](to_arcadia_string(frequency_dict_path))
        cdef TBpeDictionary* bpeDictHolder
        if self.__dictionary_type == EDictionaryType_FrequencyBased:
            dereference(self.__dictionary_holder.Get()).Save(file.Get())
        else:
            assert self.__dictionary_type == EDictionaryType_Bpe
            _ensure(bpe_path is not None, 'Specify bpe_path parameter to save bpe dictionary.')
            bpeDictHolder = dynamic_cast[TBpeDictionaryPtr](self.__dictionary_holder.Get())
            dereference(bpeDictHolder).Save(to_arcadia_string(frequency_dict_path), to_arcadia_string(bpe_path))
        return self
