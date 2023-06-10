#pragma once

#include "base.h"


#include <util/generic/array_ref.h>
#include <util/system/types.h>

enum class EMemoryType {
    Host,
    Device,
#if (CUDART_VERSION >= 10000)
    Managed,
#endif
    Cpu
};

template <class T>
EMemoryType GetPointerType(const T* ptr) {
    cudaPointerAttributes attributes;
    CUDA_SAFE_CALL(cudaPointerGetAttributes(&attributes, (void*)(ptr)));
    //TODO(noxoomo): currently don't distinguish pinned/non-pinned memory
    cudaMemoryType type;
#ifndef CUDART_VERSION
#error "CUDART_VERSION is not defined: include cuda_runtime_api.h"
#elif (CUDART_VERSION >= 10000)
    type = attributes.type;
#else
    type = attributes.memoryType;
#endif
    if (type == cudaMemoryTypeHost) {
        return EMemoryType::Host;
    } else if (type == cudaMemoryTypeDevice) {
        return EMemoryType::Device;
#if (CUDART_VERSION >= 10000)
    } else if (type == cudaMemoryTypeManaged) {
        return EMemoryType::Managed;
#endif
    } else {
        return EMemoryType::Cpu;
    }
}

template <class T>
bool IsAccessibleFromHost(const T* ptr) {
    auto type = GetPointerType(ptr);
    return
#if (CUDART_VERSION >= 10000)
        type == EMemoryType::Managed ||
#endif
        type == EMemoryType::Host || type == EMemoryType::Cpu;
}

template <class T>
class TCudaVec {
private:
    struct Inner: public TThrRefBase {
        T* Data_ = nullptr;
        ui64 Size_ = 0;
        EMemoryType Type = EMemoryType::Device;

        Inner()
            : Data_(nullptr)
            , Size_(0)
        {
        }

        Inner(ui64 size, EMemoryType type)
            : Size_(size)
            , Type(type)
        {
            if (Size_) {
                switch (type) {
                    case EMemoryType::Device: {
                        CUDA_SAFE_CALL(cudaMalloc((void**)&Data_, size * sizeof(T)));
                        break;
                    }
                    case EMemoryType::Host: {
                        CUDA_SAFE_CALL(cudaHostAlloc((void**)&Data_, size * sizeof(T), cudaHostAllocPortable));
                        break;
                    }
#if (CUDART_VERSION >= 10000)
                    case EMemoryType::Managed: {
                        CUDA_SAFE_CALL(cudaMallocManaged((void**)&Data_, size * sizeof(T)));
                        break;
                    }
#endif
                    case EMemoryType::Cpu: {
                        Data_ = new T[size];
                        break;
                    }
                }
            }
        }

        ~Inner() {
            if (Data_) {
                switch (Type) {
#if (CUDART_VERSION >= 10000)
                    case EMemoryType::Managed:
#endif
                    case EMemoryType::Device: {
                        CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaFree(Data_));
                        break;
                    }
                    case EMemoryType::Host: {
                        CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaFreeHost(Data_));
                        break;
                    }
                    case EMemoryType::Cpu: {
                        delete[] Data_;
                    }
                }
            }
        }
    };

private:
    TIntrusivePtr<Inner> Impl_;

public:

    TCudaVec(ui64 size, EMemoryType type)
        : Impl_(new Inner(size, type))
    {
    }

    explicit TCudaVec(TConstArrayRef<T> data, EMemoryType type)
        : TCudaVec(data.size(), type) {
        Write(data);
    }

    TCudaVec() {
    }

    explicit TCudaVec(std::initializer_list<T> data, EMemoryType type)
    : TCudaVec(data.size(), type) {
        TVector<T> tmp(std::move(data));
        Write(tmp);
    }

    template <EMemoryType Type>
    static TCudaVec Copy(const TCudaVec<T>& from) {
        TCudaVec result(from.Size(), Type);
        result.Write(from);
        return result;
    }

    EMemoryType MemoryType() const {
        CUDA_ENSURE(Impl_);
        return Impl_->Type;
    }

    T* Get() {
        return Impl_ ? Impl_->Data_ : nullptr;
    }

    const T* Get() const {
        return Impl_ ? Impl_->Data_ : nullptr;
    }

    ui64 Size() const {
        return Impl_ ? Impl_->Size_ : 0;
    }

    operator bool() const {
        return (bool)Impl_;
    }

    void Swap(TCudaVec& other) {
        Impl_.Swap(other.Impl_);
    }

    TArrayRef<T> Slice(ui64 offset, ui64 size) {
        return AsArrayRef().Slice(offset, size);
    }

    TArrayRef<T> Slice(ui64 offset) {
        return AsArrayRef().Slice(offset);
    }

    TConstArrayRef<T> Slice(ui64 offset) const {
        return AsArrayRef().Slice(offset);
    }

    TConstArrayRef<T> Slice(ui64 offset, ui64 size) const {
        return AsArrayRef().Slice(offset, size);
    }

    void Write(TConstArrayRef<T> src) {
        TCudaStream stream = TCudaStream::ZeroStream();
        CUDA_ENSURE(src.size() == Size(), src.size() << " ≠ " << Size());
        CUDA_ENSURE(Impl_);
        CUDA_SAFE_CALL(cudaMemcpyAsync((void*)Impl_->Data_, (const void*)src.data(), sizeof(T) * src.size(), cudaMemcpyDefault, stream));
        stream.Synchronize();
    }

    void Read(TArrayRef<T> dst) const {
        TCudaStream stream = TCudaStream::ZeroStream();
        CUDA_ENSURE(dst.size() == Size());
        CUDA_ENSURE(Impl_);
        CUDA_SAFE_CALL(cudaMemcpyAsync((void*)dst.data(), (const void*)Impl_->Data_, sizeof(T) * dst.size(), cudaMemcpyDefault, stream));
        stream.Synchronize();
    }

    void ReadAsync(TArrayRef<T> dst, TCudaStream stream) const {
        CUDA_ENSURE(dst.size() == Size());
        CUDA_ENSURE(Impl_);
        CUDA_SAFE_CALL(cudaMemcpyAsync((void*)dst.data(), (const void*)Impl_->Data_, sizeof(T) * dst.size(), cudaMemcpyDefault, stream));
    }

    operator TArrayRef<T>() {
        return AsArrayRef();
    }

    operator TConstArrayRef<T>() const {
        return AsArrayRef();
    }

    TArrayRef<T> AsArrayRef() {
        CUDA_ENSURE(*this);
        return TArrayRef<T>(Impl_->Data_, Impl_->Size_);
    }

    TArrayRef<T> AsMaybeNullptrArrayRef() {
        return Impl_ ? TArrayRef<T>(Impl_->Data_, Impl_->Size_)
                     : TArrayRef<T>((T*)nullptr, (ui64)0);
    }

    TArrayRef<const T> AsMaybeNullptrArrayRef() const {
        return Impl_ ? TArrayRef<const T>(Impl_->Data_, Impl_->Size_) : TArrayRef<const T>((const T*)nullptr, (ui64)0);
    }

    TConstArrayRef<T> AsArrayRef() const {
        CUDA_ENSURE(*this);
        return TConstArrayRef<T>(Impl_->Data_, Impl_->Size_);
    }

    void ClearAsync(const TCudaStream& stream) {
        ClearMemoryAsync(AsArrayRef(), stream);
    }
};


template <class T>
inline TCudaVec<T> MakeCudaVec(TConstArrayRef<T> data, EMemoryType type) {
    return TCudaVec<T>(data, type);
}

template <class T>
inline TCudaVec<T> MakeCudaVec(const TVector<T>& data, EMemoryType type) {
    return MakeCudaVec<T>(MakeConstArrayRef(data), type);
}

template <class T>
inline TCudaVec<T> MakeZeroVec(ui64 size, EMemoryType type) {
    if (size == 0) {
        return TCudaVec<T>();
    }
    TCudaVec<T> result(size, type);
    auto stream = TCudaStream::ZeroStream();
    result.ClearAsync(stream);
    stream.Synchronize();
    return result;
}

template <class T>
inline void MemoryCopy(TConstArrayRef<T> from, TArrayRef<T> to) {
    DeviceSynchronize();
    CUDA_ENSURE(from.size() == to.size(), from.size() << " ≠ " << to.size());
    CUDA_SAFE_CALL(cudaMemcpy((void*)to.data(), (const void*)from.data(), sizeof(T) * from.size(), cudaMemcpyDefault));
}

template <class T>
inline void MemoryCopyAsync(TConstArrayRef<T> from, TArrayRef<T> to, TCudaStream stream) {
    CUDA_ENSURE(from.size() == to.size(), from.size() << " ≠ " << to.size());
    CUDA_SAFE_CALL(cudaMemcpyAsync((void*)to.data(), (const void*)from.data(), sizeof(T) * from.size(), cudaMemcpyDefault, stream));
}


template <class T>
inline TVector<T> ReadVec(const TCudaVec<T>& data) {
    TVector<T> result(data.Size());
    data.Read(MakeArrayRef(result));
    return result;
}
