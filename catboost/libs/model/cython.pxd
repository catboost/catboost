# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *

import atexit
import six
from six import iteritems, string_types, PY3
from six.moves import range
from json import dumps, loads, JSONEncoder
from copy import deepcopy
from collections import defaultdict
import functools
import traceback
import numbers

import sys
if sys.version_info >= (3, 3):
    from collections.abc import Iterable, Sequence
else:
    from collections import Iterable, Sequence
import platform


cimport cython
from cython.operator cimport dereference, preincrement

from libc.math cimport isnan, modf
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy
from libcpp cimport bool as bool_t
from libcpp cimport nullptr
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cpython.ref cimport PyObject

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport THolder, TIntrusivePtr, MakeHolder
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui8, ui16, ui32, ui64, i32, i64
from util.string.cast cimport StrToD, TryFromString, ToString


cdef extern from "catboost/libs/model/features.h":
    cdef cppclass ENanValueTreatment "TFloatFeature::ENanValueTreatment":
        bool_t operator==(ENanValueTreatment)

    cdef ENanValueTreatment ENanValueTreatment_AsIs "TFloatFeature::ENanValueTreatment::AsIs"
    cdef ENanValueTreatment ENanValueTreatment_AsFalse "TFloatFeature::ENanValueTreatment::AsFalse"
    cdef ENanValueTreatment ENanValueTreatment_AsTrue "TFloatFeature::ENanValueTreatment::AsTrue"


cdef extern from "catboost/libs/model/model.h":
    cdef cppclass TFeaturePosition:
        int Index
        int FlatIndex

    cdef cppclass TCatFeature:
        TFeaturePosition Position
        TString FeatureId

    cdef cppclass TFloatFeature:
        bool_t HasNans
        TFeaturePosition Position
        TVector[float] Borders
        TString FeatureId

    cdef cppclass TTextFeature:
        TFeaturePosition Position
        TString FeatureId

    cdef cppclass TEmbeddingFeature:
        TFeaturePosition Position
        TString FeatureId
        int Dimension

    cdef cppclass TNonSymmetricTreeStepNode:
        ui16 LeftSubtreeDiff
        ui16 RightSubtreeDiff

    cdef cppclass IModelTreeData:
        TConstArrayRef[int] GetTreeSplits() except +ProcessException
        TConstArrayRef[int] GetTreeSizes() except +ProcessException
        TConstArrayRef[TNonSymmetricTreeStepNode] GetNonSymmetricStepNodes() except +ProcessException
        TConstArrayRef[ui32] GetNonSymmetricNodeIdToLeafId() except +ProcessException
        TConstArrayRef[double] GetLeafValues() except +ProcessException
        TConstArrayRef[double] GetLeafWeights() except +ProcessException

        void SetTreeSplits(const TVector[int]&) except +ProcessException
        void SetTreeSizes(const TVector[int]&) except +ProcessException
        void SetNonSymmetricStepNodes(const TVector[TNonSymmetricTreeStepNode]&) except +ProcessException
        void SetNonSymmetricNodeIdToLeafId(const TVector[ui32]&) except +ProcessException
        void SetLeafValues(const TVector[double]&) except +ProcessException
        void SetLeafWeights(const TVector[double]&) except +ProcessException
        THolder[IModelTreeData] Clone(ECloningPolicy policy) except +ProcessException

    cdef cppclass TModelTrees:
        int GetDimensionCount() except +ProcessException
        TConstArrayRef[TCatFeature] GetCatFeatures() except +ProcessException
        TConstArrayRef[TTextFeature] GetTextFeatures() except +ProcessException
        TConstArrayRef[TEmbeddingFeature] GetEmbeddingFeatures() except +ProcessException
        TConstArrayRef[TFloatFeature] GetFloatFeatures() except +ProcessException
        void DropUnusedFeatures() except +ProcessException
        TVector[ui32] GetTreeLeafCounts() except +ProcessException
        const THolder[IModelTreeData]& GetModelTreeData() except +ProcessException

        void ConvertObliviousToAsymmetric() except +ProcessException

    cdef cppclass TCOWTreeWrapper:
        const TModelTrees& operator*() except +ProcessException
        const TModelTrees* Get() except +ProcessException
        TModelTrees* GetMutable() except +ProcessException

    cdef cppclass TFullModel:
        TCOWTreeWrapper ModelTrees
        THashMap[TString, TString] ModelInfo

        bool_t operator==(const TFullModel& other) except +ProcessException
        bool_t operator!=(const TFullModel& other) except +ProcessException

        void Load(IInputStream* stream) except +ProcessException
        void Swap(TFullModel& other) except +ProcessException
        size_t GetTreeCount() nogil except +ProcessException
        size_t GetDimensionsCount() nogil except +ProcessException
        void Truncate(size_t begin, size_t end) except +ProcessException
        bool_t IsOblivious() except +ProcessException
        TString GetLossFunctionName() except +ProcessException
        double GetBinClassProbabilityThreshold() except +ProcessException
        TVector[TJsonValue] GetModelClassLabels() except +ProcessException
        TScaleAndBias GetScaleAndBias() except +ProcessException
        void SetScaleAndBias(const TScaleAndBias&) except +ProcessException
        void InitNonOwning(const void* binaryBuffer, size_t binarySize) except +ProcessException
        void SetEvaluatorType(EFormulaEvaluatorType evaluatorType) except +ProcessException

    cdef cppclass EModelType:
        pass

    cdef TFullModel ReadModel(const TString& modelFile, EModelType format) nogil except +ProcessException
    cdef TFullModel ReadZeroCopyModel(const void* binaryBuffer, size_t binaryBufferSize) nogil except +ProcessException
    cdef TString SerializeModel(const TFullModel& model) except +ProcessException
    cdef TFullModel DeserializeModel(const TString& serializeModelString) nogil except +ProcessException
    cdef TVector[TString] GetModelUsedFeaturesNames(const TFullModel& model) except +ProcessException
    void SetModelExternalFeatureNames(const TVector[TString]& featureNames, TFullModel* model) nogil except +ProcessException
    cdef void SaveModelBorders(const TString& file, const TFullModel& model) nogil except +ProcessException
    cdef THashMap[int, ENanValueTreatment] GetNanTreatments(const TFullModel& model) nogil except +ProcessException
