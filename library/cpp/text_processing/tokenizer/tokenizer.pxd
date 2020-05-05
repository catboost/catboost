from libcpp cimport bool as bool_t
from util.generic.vector cimport TVector
from util.generic.string cimport TString, TStringBuf


# TODO(kirillovs): this is stub. Add missing methods and move to hash.pxd
cdef extern from "util/generic/hash.h" nogil:
    cdef cppclass THashSet[T]:
        void insert(const T&) except +
        void clear() except +


cdef extern from "library/langs/langs.h" nogil:
    cdef cppclass ELanguage:
        pass


cdef extern from "library/cpp/text_processing/tokenizer/options.h" namespace "NTextProcessing::NTokenizer" nogil:
    cdef cppclass ETokenType:
        pass
    cdef cppclass ESeparatorType:
        pass
    cdef cppclass ESubTokensPolicy:
        pass
    cdef cppclass ETokenProcessPolicy:
        pass

    cdef cppclass TTokenizerOptions:
        bool_t Lowercasing
        bool_t Lemmatizing
        ETokenProcessPolicy NumberProcessPolicy
        TString NumberToken
        ESeparatorType SeparatorType
        TString Delimiter
        bool_t SplitBySet
        bool_t SkipEmpty
        THashSet[ETokenType] TokenTypes
        ESubTokensPolicy SubTokensPolicy
        TVector[ELanguage] Languages


cdef extern from "library/cpp/text_processing/tokenizer/tokenizer.h"  namespace "NTextProcessing::NTokenizer" nogil:
    cdef cppclass TTokenizer:
        TTokenizer() except +
        TTokenizer(TTokenizerOptions options) except +
        void Tokenize(TStringBuf inputString, TVector[TString]* tokens, TVector[ETokenType]* tokenTypes) except +
        TVector[TString] Tokenize(TStringBuf inputString) except +
        TTokenizerOptions GetOptions() except +
