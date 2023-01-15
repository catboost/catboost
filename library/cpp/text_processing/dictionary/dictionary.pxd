from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from libcpp cimport bool as bool_t
from util.generic.ptr cimport TIntrusivePtr
from util.generic.vector cimport TVector
from util.generic.string cimport TString, TStringBuf
from util.stream.output cimport IOutputStream
from util.system.types cimport i32, ui32, ui64


# TODO(kirillovs): move to proper util pxd definitions place
cdef extern from "util/stream/input.h" nogil:
    cdef cppclass IInputStream:
        pass

cdef extern from "util/stream/file.h" nogil:
    cdef cppclass TFileInput:
        TFileInput(...) except +

    cdef cppclass TFileOutput(IOutputStream):
        TFileOutput(...) except +


cdef extern from "library/cpp/text_processing/dictionary/types.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass EDictionaryType:
        bool_t operator==(EDictionaryType)
    cdef EDictionaryType EDictionaryType_FrequencyBased "NTextProcessing::NDictionary::EDictionaryType::FrequencyBased"
    cdef EDictionaryType EDictionaryType_Bpe "NTextProcessing::NDictionary::EDictionaryType::Bpe"

    cdef cppclass ETokenLevelType:
        pass

    cdef cppclass EEndOfWordTokenPolicy:
        pass

    cdef cppclass EEndOfSentenceTokenPolicy:
        pass

    cdef cppclass EUnknownTokenPolicy:
        pass

    ctypedef ui32 TTokenId


cdef extern from "library/cpp/text_processing/dictionary/dictionary.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass IDictionary:
        TTokenId Apply(TStringBuf token) except +

        void Apply(
            TConstArrayRef[TString] tokens,
            TVector[TTokenId]* tokensIds
        ) except +

        void Apply(
            TConstArrayRef[TString] tokens,
            TVector[TTokenId]* tokensIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) except +

        void Apply(
            TConstArrayRef[TStringBuf] tokens,
            TVector[TTokenId]* tokensIds
        ) except +

        void Apply(
            TConstArrayRef[TStringBuf] tokens,
            TVector[TTokenId]* tokensIds,
            EUnknownTokenPolicy unknownTokenPolicy
        ) except +

        ui32 Size() except +

        TString GetToken(TTokenId tokenId) except +
        TString GetTokens(TConstArrayRef[TTokenId] tokenIds, TVector[TString]* tokens) except +

        ui64 GetCount(TTokenId tokenId) except +
        TVector[TString] GetTopTokens() except +
        TVector[TString] GetTopTokens(ui32 topSize) except +
        void ClearStatsData() except +
        TTokenId GetUnknownTokenId() except +
        TTokenId GetEndOfSentenceTokenId() except +
        TTokenId GetMinUnusedTokenId() except +

        @staticmethod
        TIntrusivePtr[IDictionary] Load(IInputStream* stream) except +
        void Save(IOutputStream* stream) except +


cdef extern from "library/cpp/text_processing/dictionary/frequency_based_dictionary.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass TDictionary(IDictionary):
        TDictionary(...) except +


cdef extern from "library/cpp/text_processing/dictionary/bpe_dictionary.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass TBpeDictionary(IDictionary):
        TBpeDictionary(...) except +
        void Load(const TString& dictionaryPath, const TString& bpePath) except +
        void Save(const TString& dictionaryPath, const TString& bpePath) except +


cdef extern from "library/cpp/text_processing/dictionary/options.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass TDictionaryOptions:
        ETokenLevelType TokenLevelType
        ui32 GramOrder
        ui32 SkipStep
        TTokenId StartTokenId
        EEndOfWordTokenPolicy EndOfWordTokenPolicy
        EEndOfSentenceTokenPolicy EndOfSentenceTokenPolicy

    cdef cppclass TDictionaryBuilderOptions:
        ui64 OccurrenceLowerBound
        i32 MaxDictionarySize

    cdef cppclass TBpeDictionaryOptions:
        size_t NumUnits
        bool_t SkipUnknown


cdef extern from "library/cpp/text_processing/dictionary/dictionary_builder.h" namespace "NTextProcessing::NDictionary" nogil:
    cdef cppclass TDictionaryBuilder:
        TDictionaryBuilder(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions) except +
        TDictionaryBuilder(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions,
            i32 threadCount) except +

        void Add(...) except +
        TIntrusivePtr[TDictionary] FinishBuilding() except +
