#pragma once

#include <Python.h>

#include <library/cpp/hnsw/index/dense_vector_index.h>
#include <library/cpp/hnsw/index/dense_vector_distance.h>
#include <library/cpp/hnsw/index_builder/build_options.h>
#include <library/cpp/hnsw/index_builder/dense_vector_distance.h>
#include <library/cpp/hnsw/index_builder/dense_vector_index_builder.h>
#include <library/cpp/hnsw/index_builder/dense_vector_storage.h>
#include <library/cpp/hnsw/index_builder/index_builder.h>
#include <library/cpp/hnsw/index_builder/index_writer.h>
#include <library/cpp/hnsw/index_builder/mobius_transform.h>
#include <library/cpp/hnsw/helpers/interrupt.h>

#include <library/cpp/online_hnsw/base/build_options.h>
#include <library/cpp/online_hnsw/dense_vectors/index.h>

#include <util/generic/noncopyable.h>
#include <util/generic/buffer.h>
#include <util/generic/xrange.h>
#include <util/generic/variant.h>
#include <util/stream/buffer.h>
#include <util/system/compiler.h>


#if PY_MAJOR_VERSION < 3
inline const char* PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size) {
    Y_UNUSED(unicode, size);
    return nullptr;
}
#endif


namespace NHnsw::PythonHelpers {
    class TGilGuard : public TNonCopyable {
    public:
        TGilGuard()
            : State_(PyGILState_Ensure())
        { }

        ~TGilGuard() {
            PyGILState_Release(State_);
        }
    private:
        PyGILState_STATE State_;
    };

    void PyCheckInterrupted();
    void SetPythonInterruptHandler();
    void ResetPythonInterruptHandler();

    enum EDistance {
        DotProduct = 0,
        L1 = 1,
        L2Sqr = 2,
        PairVectorDistance = 3
    };

    template <class T>
    const char* NumpyTypeDescription();

    template <>
    const char* NumpyTypeDescription<i32>();

    template <>
    const char* NumpyTypeDescription<ui32>();

    template <>
    const char* NumpyTypeDescription<i64>();

    template <>
    const char* NumpyTypeDescription<ui64>();

    template <>
    const char* NumpyTypeDescription<float>();

    template <>
    const char* NumpyTypeDescription<double>();

    template <class T>
    PyObject* GetDistanceResultType(EDistance distance) {
        auto numpyTypeDescription = [=]() {
            switch(distance) {
                case EDistance::DotProduct: {
                    return NumpyTypeDescription<typename TDotProduct<T>::TResult>();
                    break;
                }
                case EDistance::L1: {
                    return NumpyTypeDescription<typename TL1Distance<T>::TResult>();
                    break;
                }
                case EDistance::L2Sqr: {
                    return NumpyTypeDescription<typename TL2SqrDistance<T>::TResult>();
                    break;
                }
                case EDistance::PairVectorDistance: {
                    return NumpyTypeDescription<typename TPairVectorSimilarity<T>::TResult>();
                    break;
                }
                default:
                    Y_ABORT_UNLESS(false, "Unknown distance!");
                    return "";
            }
        }();
        return Py_BuildValue("s", numpyTypeDescription);
    }

    template <class T>
    PyObject* ToPyObject(T value) {
        return PyFloat_FromDouble(value);
    }

    template <>
    PyObject* ToPyObject<i32>(i32 value);

    template <>
    PyObject* ToPyObject<ui32>(ui32 value);

    template <>
    PyObject* ToPyObject<i64>(i64 value);

    template <>
    PyObject* ToPyObject<ui64>(ui64 value);

    template <class TDistanceResult,
              class TNeighbor=NHnsw::THnswIndexBase::TNeighbor<TDistanceResult>>
    inline PyObject* ToPyObject(const TVector<TNeighbor>& neighbors) {
        PyObject* result = Py_BuildValue("[]");
        for(const TNeighbor& neigh : neighbors) {
            PyObject* obj = PyTuple_New(2);
            PyTuple_SetItem(obj, 0, ToPyObject(neigh.Id));
            PyTuple_SetItem(obj, 1, ToPyObject<TDistanceResult>(neigh.Dist));
            PyList_Append(result, obj);
            Py_DECREF(obj);
        }
        return result;
    }

    template <class T>
    inline PyObject* GetNearestNeighbors(const THnswIndexBase* index,
                                         const T* query,
                                         size_t topSize,
                                         size_t searchNeighborhoodSize,
                                         size_t distanceCalcLimit,
                                         const TDenseVectorStorage<T>* storage,
                                         EDistance distance) {
        if (distanceCalcLimit == 0) {
            distanceCalcLimit = Max<size_t>();
        }
        switch(distance) {
            case EDistance::DotProduct: {
                auto vectorDistance = TDistanceWithDimension<T, TDotProduct<T>>(TDotProduct<T>(), storage->GetDimension());
                auto neighbors = index->GetNearestNeighbors(query, topSize, searchNeighborhoodSize, distanceCalcLimit, *storage, vectorDistance);
                TGilGuard guard;
                return ToPyObject<typename TDotProduct<T>::TResult>(neighbors);
            }
            case EDistance::L1: {
                auto vectorDistance = TDistanceWithDimension<T, TL1Distance<T>>(TL1Distance<T>(), storage->GetDimension());
                auto neighbors = index->GetNearestNeighbors(query, topSize, searchNeighborhoodSize, distanceCalcLimit, *storage, vectorDistance);
                TGilGuard guard;
                return ToPyObject<typename TL1Distance<T>::TResult>(neighbors);
            }
            case EDistance::L2Sqr: {
                auto vectorDistance = TDistanceWithDimension<T, TL2SqrDistance<T>>(TL2SqrDistance<T>(), storage->GetDimension());
                auto neighbors = index->GetNearestNeighbors(query, topSize, searchNeighborhoodSize, distanceCalcLimit, *storage, vectorDistance);
                TGilGuard guard;
                return ToPyObject<typename TL2SqrDistance<T>::TResult>(neighbors);
            }
            case EDistance::PairVectorDistance: {
                auto vectorDistance = TDistanceWithDimension<T, NHnsw::TPairVectorSimilarity<T>>(NHnsw::TPairVectorSimilarity<T>(), storage->GetDimension());
                auto neighbors = index->GetNearestNeighbors(query, topSize, searchNeighborhoodSize, distanceCalcLimit, *storage, vectorDistance);
                TGilGuard guard;
                return ToPyObject<typename NHnsw::TPairVectorSimilarity<T>::TResult>(neighbors);
            }
            default:
                Y_ABORT_UNLESS(false, "Unknown distance!");
        }
    }

    template <class TDistanceResult,
              class TNeighbor=NHnsw::THnswIndexBase::TNeighbor<TDistanceResult>>
    inline void AssignResultForQuery(size_t topSize,
                                     size_t queryIdx,
                                     const TVector<TNeighbor>& neighbors,
                                     ui32* resultNeighInd, // [nQueries x topSize] array
                                     void* resultNeighDist) { // [nQueries x topSize] array, can be nullptr

        Y_ABORT_UNLESS(neighbors.size() <= topSize);

        ui32* resultNeighIndForQuery = resultNeighInd + queryIdx * topSize;
        if (resultNeighDist == nullptr) {
            for (auto neighborIdx : xrange(neighbors.size())) {
                resultNeighIndForQuery[neighborIdx] = neighbors[neighborIdx].Id;
            }
        } else {
            TDistanceResult* resultNeighDistForQuery = ((TDistanceResult*)resultNeighDist) + queryIdx * topSize;

            for (auto neighborIdx : xrange(neighbors.size())) {
                resultNeighIndForQuery[neighborIdx] = neighbors[neighborIdx].Id;
                resultNeighDistForQuery[neighborIdx] = neighbors[neighborIdx].Dist;
            }
        }
    }

    template <class T>
    inline void KNeighbors(const THnswIndexBase* index,
                           const T* queries, // [nQueries x dimension] array
                           size_t nQueries,
                           size_t topSize,
                           size_t searchNeighborhoodSize,
                           size_t distanceCalcLimit,
                           const TDenseVectorStorage<T>* storage,
                           EDistance distance,
                           ui32* resultNeighInd, // [nQueries x topSize] array
                           void* resultNeighDist) { // [nQueries x topSize] array, do not return distance if == nullptr
        size_t dimension = storage->GetDimension();
        if (distanceCalcLimit == 0) {
            distanceCalcLimit = Max<size_t>();
        }

        auto calc = [=](auto distanceInstance) {
            auto vectorDistance = TDistanceWithDimension<T, decltype(distanceInstance)>(
                distanceInstance,
                storage->GetDimension());
            for (auto queryIdx : xrange(nQueries)) {
                auto neighbors = index->GetNearestNeighbors(
                    queries + queryIdx * dimension,
                    topSize,
                    searchNeighborhoodSize,
                    distanceCalcLimit,
                    *storage,
                    vectorDistance);
                AssignResultForQuery<typename decltype(distanceInstance)::TResult>(
                    topSize,
                    queryIdx,
                    neighbors,
                    resultNeighInd,
                    resultNeighDist);
            }
        };

        switch(distance) {
            case EDistance::DotProduct: {
                calc(TDotProduct<T>());
                break;
            }
            case EDistance::L1: {
                calc(TL1Distance<T>());
                break;
            }
            case EDistance::L2Sqr: {
                calc(TL2SqrDistance<T>());
                break;
            }
            case EDistance::PairVectorDistance: {
                calc(TPairVectorSimilarity<T>());
                break;
            }
            default:
                Y_ABORT_UNLESS(false, "Unknown distance!");
        }
    }


    template <class T>
    inline TBlob BuildDenseVectorIndex(const TString& jsonOptions, const NHnsw::TDenseVectorStorage<T>* storage, EDistance distance) {
        THnswBuildOptions options = THnswBuildOptions::FromJsonString(jsonOptions);
        THnswIndexData indexData;
        switch(distance) {
            case EDistance::DotProduct:
                indexData = BuildDenseVectorIndex<T, NHnsw::TDotProduct<T>>(options, *storage, storage->GetDimension());
                break;
            case EDistance::L1:
                indexData = BuildDenseVectorIndex<T, NHnsw::TL1Distance<T>>(options, *storage, storage->GetDimension());
                break;
            case EDistance::L2Sqr:
                indexData = BuildDenseVectorIndex<T, NHnsw::TL2SqrDistance<T>>(options, *storage, storage->GetDimension());
                break;
            case EDistance::PairVectorDistance:
                indexData = BuildDenseVectorIndex<T, NHnsw::TPairVectorSimilarity<T>>(options, *storage, storage->GetDimension());
                break;
            default:
                Y_ABORT_UNLESS(false, "Unknown distance!");
        }
        TBuffer buffer;
        TBufferOutput output(buffer);
        WriteIndex(indexData, output);
        output.Finish();
        return TBlob::FromBuffer(buffer);
    }

    void SaveIndex(const TBlob& indexBlob, const TString& indexPath);
    TBlob LoadIndex(const TString& indexPath);

    template <class T>
    NHnsw::TDenseVectorStorage<float>* PyTransformMobius(const NHnsw::TDenseVectorStorage<T> *storage){
        auto transformedStorage = NHnsw::TransformMobius(*storage);
        return new NHnsw::TDenseVectorStorage<float>(std::move(transformedStorage));
    }

}


namespace NOnlineHnsw::PythonHelpers
{
    template <class T>
    class PyOnlineHnswDenseVectorIndex {
        using EDistance = NHnsw::PythonHelpers::EDistance;
        using TDotProduct = NHnsw::TDotProduct<T>;
        using TDotProductIndex = TOnlineHnswDenseVectorIndex<T, TDotProduct>;
        using TDotProductIndexHolder = THolder<TOnlineHnswDenseVectorIndex<T, TDotProduct>>;
        using TL1Distance = NHnsw::TL1Distance<T>;
        using TL1DistanceIndex = TOnlineHnswDenseVectorIndex<T, TL1Distance>;
        using TL1DistanceIndexHolder = THolder<TOnlineHnswDenseVectorIndex<T, TL1Distance>>;
        using TL2SqrDistance = NHnsw::TL2SqrDistance<T>;
        using TL2SqrDistanceIndex = TOnlineHnswDenseVectorIndex<T, TL2SqrDistance>;
        using TL2SqrDistanceIndexHolder = THolder<TOnlineHnswDenseVectorIndex<T, TL2SqrDistance>>;
        using TPairVectorSimilarity = NHnsw::TPairVectorSimilarity<T>;
        using TPairVectorSimilarityIndex = TOnlineHnswDenseVectorIndex<T, TPairVectorSimilarity>;
        using TPairVectorSimilarityIndexHolder = THolder<TOnlineHnswDenseVectorIndex<T, TPairVectorSimilarity>>;

    public:
        PyOnlineHnswDenseVectorIndex(const TString& jsonOptions, size_t dimension, EDistance distance)
            : Distance(distance)
        {
            const auto options = TOnlineHnswBuildOptions::FromJsonString(jsonOptions);
            switch(Distance) {
                case EDistance::DotProduct:
                    Index = MakeHolder<TDotProductIndex>(options, dimension);
                    break;
                case EDistance::L1:
                    Index = MakeHolder<TL1DistanceIndex>(options, dimension);
                    break;
                case EDistance::L2Sqr:
                    Index = MakeHolder<TL2SqrDistanceIndex>(options, dimension);
                    break;
                case EDistance::PairVectorDistance:
                    Index = MakeHolder<TPairVectorSimilarityIndex>(options, dimension);
                    break;
                default:
                    Y_UNREACHABLE();
            }
        }

        inline PyObject* GetNearestNeighbors(const T* query, size_t topSize){
            if(topSize == 0)
                topSize = Max<size_t>();
            switch(Distance) {
                case EDistance::DotProduct: {
                    const auto& indexImpl = std::get<TDotProductIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighbors(query, topSize);
                    return NHnsw::PythonHelpers::ToPyObject<typename TDotProduct::TResult>(neighbors);
                }
                case EDistance::L1: {
                    const auto& indexImpl = std::get<TL1DistanceIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighbors(query, topSize);
                    return NHnsw::PythonHelpers::ToPyObject<typename TL1Distance::TResult>(neighbors);
                }
                case EDistance::L2Sqr: {
                    const auto& indexImpl = std::get<TL2SqrDistanceIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighbors(query, topSize);
                    return NHnsw::PythonHelpers::ToPyObject<typename TL2SqrDistance::TResult>(neighbors);
                }
                case EDistance::PairVectorDistance: {
                    const auto& indexImpl = std::get<TPairVectorSimilarityIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighbors(query, topSize);
                    return NHnsw::PythonHelpers::ToPyObject<typename TPairVectorSimilarity::TResult>(neighbors);
                }
                default:
                    Y_UNREACHABLE();
            }
        }

        inline PyObject* GetNearestNeighborsAndAddItem(const T* query){
            switch(Distance) {
                case EDistance::DotProduct: {
                    auto& indexImpl = std::get<TDotProductIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighborsAndAddItem(query);
                    return NHnsw::PythonHelpers::ToPyObject<typename TDotProduct::TResult>(neighbors);
                }
                case EDistance::L1: {
                    auto& indexImpl = std::get<TL1DistanceIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighborsAndAddItem(query);
                    return NHnsw::PythonHelpers::ToPyObject<typename TL1Distance::TResult>(neighbors);
                }
                case EDistance::L2Sqr: {
                    auto& indexImpl = std::get<TL2SqrDistanceIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighborsAndAddItem(query);
                    return NHnsw::PythonHelpers::ToPyObject<typename TL2SqrDistance::TResult>(neighbors);
                }
                case EDistance::PairVectorDistance: {
                    auto& indexImpl = std::get<TPairVectorSimilarityIndexHolder>(Index);
                    auto neighbors = indexImpl->GetNearestNeighborsAndAddItem(query);
                    return NHnsw::PythonHelpers::ToPyObject<typename TPairVectorSimilarity::TResult>(neighbors);
                }
                default:
                    Y_UNREACHABLE();
            }
        }

        inline void AddItem(const T* item) {
            switch(Distance) {
                case EDistance::DotProduct: {
                    auto& indexImpl = std::get<TDotProductIndexHolder>(Index);
                    indexImpl->GetNearestNeighborsAndAddItem(item);
                    break;
                }
                case EDistance::L1: {
                    auto& indexImpl = std::get<TL1DistanceIndexHolder>(Index);
                    indexImpl->GetNearestNeighborsAndAddItem(item);
                    break;
                }
                case EDistance::L2Sqr: {
                    auto& indexImpl = std::get<TL2SqrDistanceIndexHolder>(Index);
                    indexImpl->GetNearestNeighborsAndAddItem(item);
                    break;
                }
                case EDistance::PairVectorDistance: {
                    auto& indexImpl = std::get<TPairVectorSimilarityIndexHolder>(Index);
                    indexImpl->GetNearestNeighborsAndAddItem(item);
                    break;
                }
                default:
                    Y_UNREACHABLE();
            }
        }

        inline const T* GetItem(size_t id) {
            return std::visit([id](const auto& indexImpl) -> const T* {
                return indexImpl->GetItem(id);
            }, Index);
        }

        inline size_t GetNumItems() {
            return std::visit([](const auto& indexImpl) -> size_t {
                return indexImpl->GetNumItems();
            }, Index);
        }

    private:
        EDistance Distance;
        std::variant<
            TDotProductIndexHolder,
            TL1DistanceIndexHolder,
            TL2SqrDistanceIndexHolder,
            TPairVectorSimilarityIndexHolder
        > Index;
    };
}
