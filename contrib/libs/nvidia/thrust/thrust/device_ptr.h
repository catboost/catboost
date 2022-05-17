/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file device_ptr.h
 *  \brief A pointer to a variable which resides memory accessible to devices.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/memory.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_management_classes Memory Management Classes
 *  \ingroup memory_management
 *  \{
 */

// forward declarations
template<typename T> class device_reference;

/*! \p device_ptr stores a pointer to an object allocated in device memory. This type
 *  provides type safety when dispatching standard algorithms on ranges resident in
 *  device memory.
 *
 *  \p device_ptr has pointer semantics: it may be dereferenced safely from the host and
 *  may be manipulated with pointer arithmetic.
 *
 *  \p device_ptr can be created with the functions device_malloc, device_new, or
 *  device_pointer_cast, or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p device_ptr may be obtained by either its <tt>get</tt>
 *  method or the \p raw_pointer_cast free function.
 *
 *  \note \p device_ptr is not a smart pointer; it is the programmer's responsibility to
 *  deallocate memory pointed to by \p device_ptr.
 *
 *  \see device_malloc
 *  \see device_new
 *  \see device_pointer_cast
 *  \see raw_pointer_cast
 */
template<typename T>
  class device_ptr
    : public thrust::pointer<
               T,
               thrust::device_system_tag,
               thrust::device_reference<T>,
               thrust::device_ptr<T>
             >
{
  private:
    typedef thrust::pointer<
      T,
      thrust::device_system_tag,
      thrust::device_reference<T>,
      thrust::device_ptr<T>
    > super_t;

  public:
    /*! \p device_ptr's null constructor initializes its raw pointer to \c 0.
     */
    __host__ __device__
    device_ptr() : super_t() {}

    #if THRUST_CPP_DIALECT >= 2011
    // NOTE: This is needed so that Thrust smart pointers can be used in
    // `std::unique_ptr`.
    __host__ __device__
    device_ptr(decltype(nullptr)) : super_t(nullptr) {}
    #endif

    /*! \p device_ptr's copy constructor is templated to allow copying to a
     *  <tt>device_ptr<const T></tt> from a <tt>T *</tt>.
     *
     *  \param ptr A raw pointer to copy from, presumed to point to a location in
     *         device memory.
     */
    template<typename OtherT>
    __host__ __device__
    explicit device_ptr(OtherT *ptr) : super_t(ptr) {}

    /*! \p device_ptr's copy constructor allows copying from another device_ptr with related type.
     *  \param other The \p device_ptr to copy from.
     */
    template<typename OtherT>
    __host__ __device__
    device_ptr(const device_ptr<OtherT> &other) : super_t(other) {}

    /*! \p device_ptr's assignment operator allows assigning from another \p device_ptr with related type.
     *  \param other The other \p device_ptr to copy from.
     *  \return <tt>*this</tt>
     */
    template<typename OtherT>
    __host__ __device__
    device_ptr &operator=(const device_ptr<OtherT> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    #if THRUST_CPP_DIALECT >= 2011
    // NOTE: This is needed so that Thrust smart pointers can be used in
    // `std::unique_ptr`.
    __host__ __device__
    device_ptr& operator=(decltype(nullptr))
    {
      super_t::operator=(nullptr);
      return *this;
    }
    #endif

// declare these members for the purpose of Doxygenating them
// they actually exist in a derived-from class
#if 0
    /*! This method returns this \p device_ptr's raw pointer.
     *  \return This \p device_ptr's raw pointer.
     */
    __host__ __device__
    T *get(void) const;
#endif // end doxygen-only members
}; // end device_ptr

// declare these methods for the purpose of Doxygenating them
// they actually are defined for a derived-from class
#if 0
/*! Writes to an output stream the value of a \p device_ptr's raw pointer.
 *
 *  \param os The output stream.
 *  \param p The \p device_ptr to output.
 *  \return os.
 */
template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os, const device_ptr<T> &p);
#endif

/*! \}
 */


/*!
 *  \addtogroup memory_management_functions Memory Management Functions
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_pointer_cast creates a device_ptr from a raw pointer which is presumed to point
 *  to a location in device memory.
 *
 *  \param ptr A raw pointer, presumed to point to a location in device memory.
 *  \return A device_ptr wrapping ptr.
 */
template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(T *ptr);

/*! This version of \p device_pointer_cast creates a copy of a device_ptr from another device_ptr.
 *  This version is included for symmetry with \p raw_pointer_cast.
 *
 *  \param ptr A device_ptr.
 *  \return A copy of \p ptr.
 */
template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr);

/*! \}
 */

THRUST_NAMESPACE_END

#include <thrust/detail/device_ptr.inl>
#include <thrust/detail/raw_pointer_cast.h>
