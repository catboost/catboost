#pragma once
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <cub/device/device_scan.cuh>

namespace NKernel {

    template <typename T,
        typename TOffsetT = ptrdiff_t>
    class TScanBitIterator {
    public:
        using TValueType = int;
        // Required iterator traits
        typedef TScanBitIterator self_type;              ///< My own type
        typedef TOffsetT
            difference_type;        ///< Type to express the result of subtracting one iterator from another
        typedef TValueType value_type;             ///< The type of the element the iterator can point to
        typedef TValueType reference;              ///< The type of a reference to an element the iterator can point to
        typedef std::random_access_iterator_tag iterator_category;      ///< The iterator category
        typedef T* pointer;


    private:
        const T* Bins;
        i32 Bit;
    public:

        /// Constructor
        __host__ __device__
        __forceinline__ TScanBitIterator(
            const T* src,
            i32 bit)
            : Bins(src)
              , Bit(bit) {

        }

        /// Postfix increment
        __host__ __device__
        __forceinline__ self_type
        operator++(int) {
            self_type retval = *this;
            Bins++;
            return retval;
        }

        /// Prefix increment
        __host__ __device__
        __forceinline__ self_type
        operator++() {
            Bins++;
            return *this;
        }

        /// Indirection
        __host__ __device__
        __forceinline__ TValueType
        operator*() const {
            return (static_cast<ui32>(Ldg(Bins)) >> Bit) & 1;
        }

        /// Addition
        template <typename Distance>
        __host__ __device__
        __forceinline__ self_type
        operator+(Distance n) const {
            self_type retval(Bins + n, Bit);
            return retval;
        }

        /// Addition assignment
        template <typename Distance>
        __host__ __device__
        __forceinline__ self_type& operator+=(Distance n) {
            Bins += n;
            return *this;
        }

        /// Subtraction
        template <typename Distance>
        __host__ __device__
        __forceinline__ self_type
        operator-(Distance n) const {
            self_type retval(Bins - n, Bit);
            return retval;
        }

        /// Subtraction assignment
        template <typename Distance>
        __host__ __device__
        __forceinline__ self_type&
        operator-=(Distance n) {
            Bins -= n;
            return *this;
        }

        /// Distance
        __host__ __device__
        __forceinline__ difference_type
        operator-(self_type other) const {
            return Bins - other.Bins;
        }

        /// Array subscript
        template <typename Distance>
        __host__ __device__
        __forceinline__ TValueType
        operator[](Distance n) const {
            static_assert(sizeof(T) <= sizeof(ui32), "for <= 32 bit keys only");
            return (static_cast<ui32>(Ldg(Bins + n)) >> Bit) & 1;
        }

        /// Equal to
        __host__ __device__
        __forceinline__ bool operator==(const self_type& rhs) {
            return Bins == rhs.Bins;
        }

        /// Not equal to
        __host__ __device__
        __forceinline__ bool operator!=(const self_type& rhs) {
            return !TScanBitIterator::operator==(rhs);
        }
    };


    template <typename T, class V, int N, int BlockSize>
    __global__ void ReorderOneBitImpl(
        const T* __restrict__ tempKeys,
        const V* __restrict__ tempValues,
        int* offsets,
        int bit,
        T* __restrict__ keys,
        V* __restrict__ values,
        i32 size) {
//
        static_assert(sizeof(T) <= sizeof(ui32), "implemented for 32-bit and less bit keys only");

        const i32 i = N * blockIdx.x * BlockSize + threadIdx.x;
        const int totalOnes = __ldg(offsets + size - 1) + ((static_cast<ui32>(Ldg(tempKeys + size - 1)) >> bit) & 1);
        const int totalZeros = size - totalOnes;

        T key[N];

        int onesBefore[N];
        int zeroesBefore[N];
        int offset[N];
        V tempVals[N];

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            const int idx = i + k * BlockSize;
            if (idx < size) {
                onesBefore[k] = idx < size ? __ldg(offsets + idx) : 0;
                key[k] = Ldg(tempKeys + idx);
                tempVals[k] = __ldg(tempValues + idx);
                zeroesBefore[k] = idx - onesBefore[k];
            }
        }

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            const int idx = i + k * BlockSize;
            if (idx < size) {
                bool isZero = ((key[k] >> bit) & 1) == 0;
                offset[k] = isZero ? zeroesBefore[k] : (totalZeros + onesBefore[k]);
            }
        }

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            const int idx = i + k * BlockSize;
            if (idx < size) {
                keys[offset[k]] = key[k];
                values[offset[k]] = tempVals[k];
            }
        }
    }





}



