#include "scan.cuh"
#include "kernel_helpers.cuh"
#include "index_wrapper.cuh"
#include <cub/device/device_scan.cuh>

#include <type_traits>

namespace NKernel
{

    struct TNonNegativeSegmentedSum
    {
        template <typename T>
        __host__ __device__ __forceinline__ T operator()(const T& left, const T& right) const
        {
            const bool leftFlag = ExtractSignBit(left);
            const bool rightFlag = ExtractSignBit(right);
            const bool newFlag = leftFlag | rightFlag;
            const T resultValue = rightFlag ? abs(right) : abs(right) + abs(left);
            return OrSignBit(resultValue, newFlag);
        }
    };

    struct TSegmentedSum
    {
        template <typename T>
        __host__ __device__ __forceinline__ TPair<ui32, T> operator()(const TPair<ui32, T>& left, const TPair<ui32, T>& right) const {
            const bool leftFlag =  left.First;
            const bool rightFlag = right.First;
            const bool newFlag = leftFlag || rightFlag;
            const T resultValue = rightFlag ? right.Second : left.Second + right.Second;
            return {newFlag, resultValue};
        }
    };

    //output iterator for segmented scan + scatter with mask routine
    //class based on cache-modified-output iterator from cub
    template <cub::CacheStoreModifier MODIFIER,
             typename TValueType,
             typename TOffsetType = ptrdiff_t,
             bool Inclusive = false>
    class TNonNegativeSegmentedScanOutputIterator
    {
    private:
        // Proxy object
        struct TReference
        {
            TValueType* Ptr;
            const ui32* __restrict  Index;
            const ui32* __restrict  End;

            /// Constructor
            __host__ __device__ __forceinline__ TReference(TValueType* ptr,
                                                           const ui32* __restrict  index,
                                                           const ui32* __restrict  end)
                    : Ptr(ptr)
                    , Index(index)
                    , End(end) {
            }

            /// Assignment
            __device__ __forceinline__ TValueType operator=(TValueType val) {
                if (Inclusive) {
                    TIndexWrapper indexWrapper(Index[0]);
                    TValueType outputValue = abs(val);
                    Ptr[indexWrapper.Index()] = outputValue;
                } else {
                    if ((Index + 1) != End)
                    {
                        TIndexWrapper indexWrapper(Index[1]);
                        TValueType outputValue = indexWrapper.IsSegmentStart() ? 0 : abs(val);
                        Ptr[indexWrapper.Index()] = outputValue;
                    }
                }
                return val;
            }
        };

    private:
        TValueType* Ptr;
        const ui32* __restrict Index;
        const ui32* __restrict End;
    public:
        // Required iterator traits
        typedef TNonNegativeSegmentedScanOutputIterator self_type;              ///< My own type
        typedef TOffsetType difference_type;        ///< Type to express the result of subtracting one iterator from another
        typedef TValueType value_type;             ///< The type of the element the iterator can point to
        typedef TValueType* pointer;                ///< The type of a pointer to an element the iterator can point to
        typedef TReference reference;              ///< The type of a reference to an element the iterator can point to

        typedef std::random_access_iterator_tag iterator_category;

    public:

        /// Constructor
        template <class TQualifiedValueType>
        __host__ __device__ __forceinline__ TNonNegativeSegmentedScanOutputIterator(TQualifiedValueType* ptr,
                                                                                    const ui32* __restrict index,
                                                                                    const ui32* __restrict end
        )
        : Ptr(const_cast<typename std::remove_cv<TQualifiedValueType>::type*>(ptr))
        , Index(index)
        , End(end) {
        }

        /// Postfix increment
        __host__ __device__ __forceinline__ self_type operator++(int) {
            self_type retval = *this;
            Index++;
            return retval;
        }


        /// Prefix increment
        __host__ __device__ __forceinline__ self_type operator++() {
            Index++;
            return *this;
        }

        /// Indirection
        __host__ __device__ __forceinline__ reference operator*() const
        {
            return TReference(Ptr, Index, End);
        }

        /// Addition
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type operator+(TDistance n) const
        {
            self_type retval(Ptr, Index + n, End);
            return retval;
        }

        /// Addition assignment
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type& operator+=(TDistance n) {
            Index += n;
            return *this;
        }

        /// Subtraction
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type operator-(TDistance n) const {
            self_type retval(Ptr, Index - n, End);
            return retval;
        }

        /// Subtraction assignment
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type& operator-=(TDistance n) {
            Index -= n;
            return *this;
        }

        /// TDistance
        __host__ __device__
        __forceinline__ difference_type operator-(self_type other) const {
            return Index - other.Index;
        }

        /// Array subscript
        template <typename TDistance>
        __host__ __device__ __forceinline__ reference operator[](TDistance n) const {
            return TReference(Ptr, Index + n, End);
        }

        /// Equal to
        __host__ __device__ __forceinline__ bool operator==(const self_type& rhs) {
            return (Index == rhs.Index) && (Ptr == rhs.Ptr);
        }

        /// Not equal to
        __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs) {
            return !TNonNegativeSegmentedScanOutputIterator::operator==(rhs);
        }
    };


    //output iterator for segmented scan + scatter with mask routine
    //class based on cache-modified-output iterator from cub
    template <typename T,
             bool Inclusive = false,
             typename TOffsetType = ptrdiff_t>
    class TSegmentedScanOutputIterator
    {
    private:
        using TValueType = TPair<ui32, T>;
        // Proxy object
        struct TReference
        {
            T* Ptr;
            T* End;

            /// Constructor
            __host__ __device__ __forceinline__ TReference(T* ptr,
                                                           T* end)
                : Ptr(ptr)
                , End(end) {
            }

            /// Assignment
            __device__ __forceinline__ TValueType operator=(TValueType val) {
                if (Inclusive) {
                    Ptr[0] = val.Second;
                } else {
                    if ((Ptr + 1) != End)
                    {
                        Ptr[1] = val.Second;
                    }
                }
                return val;
            }
        };

    private:
        T* Ptr;
        T* End;
    public:
        // Required iterator traits
        typedef TSegmentedScanOutputIterator self_type;              ///< My own type
        typedef TOffsetType difference_type;        ///< Type to express the result of subtracting one iterator from another
        typedef TValueType value_type;             ///< The type of the element the iterator can point to
        typedef TValueType* pointer;                ///< The type of a pointer to an element the iterator can point to
        typedef TReference reference;              ///< The type of a reference to an element the iterator can point to

        typedef std::random_access_iterator_tag iterator_category;

    public:

        /// Constructor
        template <class TQualifiedValueType>
        __host__ __device__ __forceinline__ TSegmentedScanOutputIterator(TQualifiedValueType* ptr,
                                                                         TQualifiedValueType* end)
                : Ptr(const_cast<typename std::remove_cv<TQualifiedValueType>::type*>(ptr))
                , End(const_cast<typename std::remove_cv<TQualifiedValueType>::type*>(end)) {
        }

        /// Postfix increment
        __host__ __device__ __forceinline__ self_type operator++(int) {
            self_type retval = *this;
            Ptr++;
            return retval;
        }


        /// Prefix increment
        __host__ __device__ __forceinline__ self_type operator++() {
            Ptr++;
            return *this;
        }

        /// Indirection
        __host__ __device__ __forceinline__ reference operator*() const
        {
            return TReference(Ptr, End);
        }

        /// Addition
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type operator+(TDistance n) const
        {
            self_type retval(Ptr + n, End);
            return retval;
        }

        /// Addition assignment
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type& operator+=(TDistance n) {
            Ptr += n;
            return *this;
        }

        /// Subtraction
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type operator-(TDistance n) const {
            self_type retval(Ptr - n, End);
            return retval;
        }

        /// Subtraction assignment
        template <typename TDistance>
        __host__ __device__ __forceinline__ self_type& operator-=(TDistance n) {
            Ptr -= n;
            return *this;
        }

        /// TDistance
        __host__ __device__
        __forceinline__ difference_type operator-(self_type other) const {
            return Ptr - other.Ptr;
        }

        /// Array subscript
        template <typename TDistance>
        __host__ __device__ __forceinline__ reference operator[](TDistance n) const {
            return TReference(Ptr + n, End);
        }

        /// Equal to
        __host__ __device__ __forceinline__ bool operator==(const self_type& rhs) {
            return (Ptr == rhs.Ptr) && (End == rhs.End);
        }

        /// Not equal to
        __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs) {
            return !TSegmentedScanOutputIterator::operator==(rhs);
        }
    };




    template <typename T,
              typename TOffsetT = ptrdiff_t>
    class TSegmentedScanInputIterator
    {
    public:
        using TValueType = TPair<ui32, T>;
        // Required iterator traits
        typedef TSegmentedScanInputIterator         self_type;              ///< My own type
        typedef TOffsetT                            difference_type;        ///< Type to express the result of subtracting one iterator from another
        typedef TValueType                          value_type;             ///< The type of the element the iterator can point to
        typedef TValueType                          reference;              ///< The type of a reference to an element the iterator can point to
        typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
        typedef T*                                  pointer;
    private:
        const T*   Src;
        const ui32* Flags;
        ui32 FlagMask;
    public:

        /// Constructor
        __host__ __device__ __forceinline__  TSegmentedScanInputIterator(const T* src,
                                                                         const ui32* flags,
                                                                         ui32 flagMask)
        : Src(src)
        , Flags(flags)
        , FlagMask(flagMask) {

        }

        /// Postfix increment
        __host__ __device__ __forceinline__ self_type operator++(int)
        {
            self_type retval = *this;
            Src++;
            Flags++;
            return retval;
        }

        /// Prefix increment
        __host__ __device__ __forceinline__ self_type operator++()
        {
            Src++;
            Flags++;
            return *this;
        }

        /// Indirection
        __host__ __device__ __forceinline__ reference operator*() const
        {
            bool flag = Flags[0] & FlagMask;
            return {flag, Src[0]};
        }

        /// Addition
        template <typename Distance>
        __host__ __device__ __forceinline__ self_type operator+(Distance n) const
        {
            self_type retval(Src + n, Flags + n, FlagMask);
            return retval;
        }

        /// Addition assignment
        template <typename Distance>
        __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
        {
            Src += n;
            Flags += n;
            return *this;
        }

        /// Subtraction
        template <typename Distance>
        __host__ __device__ __forceinline__ self_type operator-(Distance n) const
        {
            self_type retval(Src - n, Flags - n, FlagMask);
            return retval;
        }

        /// Subtraction assignment
        template <typename Distance>
        __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
        {
            Src -= n;
            Flags -= n;
            return *this;
        }

        /// Distance
        __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
        {
            return Src - other.Src;
        }

        /// Array subscript
        template <typename Distance>
        __host__ __device__ __forceinline__ reference operator[](Distance n) const
        {
            bool flag = Flags[n] & FlagMask;
            return {flag, Src[n]};
        }

        /// Structure dereference
        __host__ __device__ __forceinline__ pointer operator->()
        {
            return Src;
        }

        /// Equal to
        __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
        {
            return (Src == rhs.Src && Flags == rhs.Flags && FlagMask == rhs.FlagMask);
        }

        /// Not equal to
        __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
        {
            return !TSegmentedScanInputIterator::operator==(rhs);
        }
    };
}
