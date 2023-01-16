// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdint>
#include <cstring>
#include <malloc.h>
#include <windows.h> // For RtlPcToFileHeader function

struct EHCatchableType {
  uint32_t properties;
  int32_t type_info;
  uint32_t non_virtual_adjustment;
  uint32_t offset_to_virtual_base_ptr;
  uint32_t virtual_base_table_index;
  uint32_t size;
  int32_t copy_function;
};

struct EHCatchableTypeArray {
  uint32_t catchable_types;
  // It is variable size but we only need the first element of this array
  int32_t array_of_catchable_types[1];
};

struct EHThrowInfo {
  uint32_t attributes;
  int32_t unwind;
  int32_t forward_compat;
  int32_t catchable_type_array;
};

struct EHParameters {
  uint32_t magic_number;
  void* exception_object;
  EHThrowInfo* throw_info;
#ifdef _M_AMD64
  uintptr_t throw_image_base;
#endif
};

struct EHExceptionRecord {
  uint32_t exception_code;
  uint32_t exception_flags;
  void* exception_record;
  void* exception_address;
  uint32_t number_of_parameters;
  EHParameters parameters;
};

// defined in vcruntime<ver>.dll
extern "C" EHExceptionRecord** __current_exception();

// This is internal compiler definition for MSVC but not for clang.
// We use our own EHThrowInfo because _ThrowInfo doesn't match actual
// compiler-generated structures in 64-bit mode.
#ifdef __clang__
struct _ThrowInfo;
// defined in vcruntime<ver>.dll
_LIBCPP_NORETURN extern "C" void __stdcall _CxxThrowException(
    void* __exc, _ThrowInfo* __throw_info);
#endif

namespace {
struct ExceptionPtr {
  void* exception_object;
  const EHThrowInfo* throw_info;
  std::atomic<size_t> counter;
#ifdef _M_AMD64
  PVOID image_base;
#endif
  template <class T>
  T convert(int32_t offset) {
#ifdef _M_AMD64
    uintptr_t value = reinterpret_cast<uintptr_t>(image_base) +
                      static_cast<uintptr_t>(offset);
#else
    uintptr_t value = static_cast<uintptr_t>(offset);
#endif
    T res;
    static_assert(
        sizeof(value) == sizeof(res),
        "Can only convert to pointers or pointers to member functions");
    memcpy(&res, &value, sizeof(value));
    return res;
  }

  void copy(void* dst, const void* src, const EHCatchableType* exc_type) {
    struct Temp {};
    constexpr uint32_t virtual_base = 4;
    if (exc_type->copy_function == 0) {
      memcpy(dst, src, exc_type->size);
    } else if (exc_type->properties & virtual_base) {
      auto copy_constructor =
          convert<void (Temp::*)(const void*, int)>(exc_type->copy_function);
      ((Temp*)dst->*copy_constructor)(src, 1);
    } else {
      auto copy_constructor =
          convert<void (Temp::*)(const void*)>(exc_type->copy_function);
      ((Temp*)dst->*copy_constructor)(src);
    }
  }

  EHCatchableType* exception_type() {
    return convert<EHCatchableType*>(
        convert<EHCatchableTypeArray*>(throw_info->catchable_type_array)
            ->array_of_catchable_types[0]);
  }

  ExceptionPtr(const void* exception_object_, const EHThrowInfo* throw_info_)
      : exception_object(nullptr), throw_info(throw_info_), counter(1) {
#ifdef _M_AMD64
    RtlPcToFileHeader(
        reinterpret_cast<PVOID>(const_cast<EHThrowInfo*>(throw_info)),
        &image_base);
#endif
    EHCatchableType* exc_type = exception_type();
    this->exception_object = malloc(exc_type->size);
    if (this->exception_object == nullptr) {
      throw std::bad_alloc();
    }
    copy(exception_object, exception_object_, exc_type);
  }

  ~ExceptionPtr() {
    if (throw_info->unwind && exception_object) {
      struct Temp {};
      auto destructor = convert<void (Temp::*)()>(throw_info->unwind);
      ((Temp*)exception_object->*destructor)();
    }
    free(exception_object);
  }

  // _bad_alloc_storage must be initialized before bad_alloc, so we declare and define it first.
  static std::bad_alloc _bad_alloc_storage;
  static ExceptionPtr bad_alloc;
  //static ExceptionPtr bad_exception;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Waddress-of-temporary"
#endif

std::bad_alloc ExceptionPtr::_bad_alloc_storage;

ExceptionPtr ExceptionPtr::bad_alloc(
    &ExceptionPtr::_bad_alloc_storage,
    reinterpret_cast<const EHThrowInfo*>(__GetExceptionInfo(ExceptionPtr::_bad_alloc_storage)));

/* ExceptionPtr
ExceptionPtr::bad_exception(&std::bad_exception(),
                            reinterpret_cast<const EHThrowInfo*>(
                                __GetExceptionInfo(std::bad_exception()))); */

#ifdef __clang__
#pragma clang diagnostic pop
#endif

} // namespace

namespace std {

exception_ptr::exception_ptr(const exception_ptr& __other) noexcept
    : __ptr_(__other.__ptr_) {
  if (__ptr_) {
    reinterpret_cast<ExceptionPtr*>(__ptr_)->counter.fetch_add(1);
  }
}

exception_ptr& exception_ptr::
operator=(const exception_ptr& __other) noexcept {
  auto before = __ptr_;
  __ptr_ = __other.__ptr_;
  if (__ptr_) {
    reinterpret_cast<ExceptionPtr*>(__ptr_)->counter.fetch_add(1);
  }
  if (before) {
    if (reinterpret_cast<ExceptionPtr*>(before)->counter.fetch_sub(1) == 1) {
      delete reinterpret_cast<ExceptionPtr*>(before);
    }
  }
  return *this;
}

exception_ptr::~exception_ptr() noexcept {
  if (__ptr_) {
    if (reinterpret_cast<ExceptionPtr*>(__ptr_)->counter.fetch_sub(1) == 1) {
      delete reinterpret_cast<ExceptionPtr*>(__ptr_);
    }
  }
}

exception_ptr __copy_exception_ptr(void* exception_object,
                                   const void* throw_info) {
  ExceptionPtr* ptr;
  try {
    ptr = new ExceptionPtr(exception_object,
                           reinterpret_cast<const EHThrowInfo*>(throw_info));
  } catch (const std::bad_alloc&) {
    ptr = &ExceptionPtr::bad_alloc;
    ptr->counter.fetch_add(1);
  } catch (...) {
    //ptr = &ExceptionPtr::bad_exception;
    //ptr->counter.fetch_add(1);
    std::terminate();
  }
  exception_ptr res;
  memcpy(&res, &ptr, sizeof(ptr));
  return res;
}

exception_ptr current_exception() noexcept {
  EHExceptionRecord** record = __current_exception();
  if (*record && !std::uncaught_exception()) {
    return __copy_exception_ptr((*record)->parameters.exception_object,
                                (*record)->parameters.throw_info);
  }
  return exception_ptr();
}

_LIBCPP_NORETURN
void rethrow_exception(exception_ptr p) {
  if (!p) {
    throw std::bad_exception();
  }
  ExceptionPtr* exc_ptr = reinterpret_cast<ExceptionPtr*>(p.__ptr_);
  EHCatchableType* exc_type = exc_ptr->exception_type();
  // _CxxThrowException doesn't call free on exception object so we must
  // allocate it on the stack.
  void* dst = _alloca(exc_type->size);
  exc_ptr->copy(dst, exc_ptr->exception_object, exc_type);
  auto throw_info = reinterpret_cast<_ThrowInfo*>(
      const_cast<EHThrowInfo*>(exc_ptr->throw_info));
  // For some reason clang doesn't call p destructor during unwinding.
  // So we must clear it ourselves.
  p = nullptr;
  _CxxThrowException(dst, throw_info);
}

nested_exception::nested_exception() noexcept : __ptr_(current_exception()) {}

nested_exception::~nested_exception() noexcept {}

_LIBCPP_NORETURN
void nested_exception::rethrow_nested() const {
  if (__ptr_ == nullptr)
    terminate();
  rethrow_exception(__ptr_);
}

} // namespace std
