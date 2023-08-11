/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_ITER_H_
#define FLATBUFFERS_ITER_H_

#include "flatbuffers.h"
#include <optional>

/// @file
namespace yandex {
namespace maps {
namespace flatbuffers_iter {

#define FLATBUFFERS_FILE_IDENTIFIER_LENGTH 4

using flatbuffers::uoffset_t;
using flatbuffers::soffset_t;
using flatbuffers::voffset_t;
using flatbuffers::EndianScalar;

// Wrapper for uoffset_t to allow safe template specialization.
template<typename T> struct Offset {
  uoffset_t o;
  Offset() : o(0) {}
  Offset(uoffset_t _o) : o(_o) {}
  Offset<void> Union() const { return Offset<void>(o); }
};

template<typename Iter>
inline bool hasContiguous(const Iter& spot, uoffset_t length)
{
    return spot.hasContiguous(length);
}

inline bool hasContiguous(const uint8_t* /* spot */, uoffset_t /* length */)
{
    return true;
}

template <typename Iter>
inline const uint8_t* getRawPointer(const Iter& spot)
{
    return spot.rawPointer();
}

inline const uint8_t* getRawPointer(const uint8_t* spot)
{
    return spot;
}

template<typename T, typename Iter>
typename std::enable_if<sizeof(T) == 1, T>::type extractValue(const Iter& spot)
{
    typename std::remove_cv<T>::type ret;
    std::memcpy(&ret, getRawPointer(spot), 1);
    return ret;
}

template<typename T, typename Iter>
typename std::enable_if<sizeof(T) != 1, T>::type extractValue(const Iter& spot)
{
    if (hasContiguous(spot, sizeof(T))) {
        typename std::remove_cv<T>::type ret;
        std::memcpy(&ret, getRawPointer(spot), sizeof(T));
        return ret;
    }
    Iter itr = spot;
    alignas(T) uint8_t buf[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        buf[i] = *itr;
        ++itr;
    }
    return *reinterpret_cast<T*>(buf);
}

template<typename T, typename Iter> T ReadScalar(Iter p) {
  return EndianScalar(extractValue<T>(p));
}

// When we read serialized data from memory, in the case of most scalars,
// we want to just read T, but in the case of Offset, we want to actually
// perform the indirection and return a pointer.
// The template specialization below does just that.
// It is wrapped in a struct since function templates can't overload on the
// return type like this.
// The typedef is for the convenience of callers of this function
// (avoiding the need for a trailing return decltype)
template<typename T> struct IndirectHelper {
  typedef T return_type;
  typedef T mutable_return_type;
  static const size_t element_stride = sizeof(T);
  template<typename Iter>
  static return_type Read(const Iter& p, uoffset_t i) {
    return i ? EndianScalar(extractValue<return_type>(p+sizeof(return_type)*i)) : EndianScalar(extractValue<return_type>(p));
  }
};
template<typename T> struct IndirectHelper<Offset<T>> {
  typedef std::optional<T> return_type;
  typedef std::optional<T> mutable_return_type;
  static const size_t element_stride = sizeof(uoffset_t);
  template<typename Iter>
  static return_type Read(Iter p, uoffset_t i) {
    p += i * sizeof(uoffset_t);
    return return_type(T(p + ReadScalar<uoffset_t>(p)));
  }
};
template<typename T> struct IndirectHelper<const T *> {
};


// An STL compatible iterator implementation for Vector below, effectively
// calling Get() for every element.
template<typename T, typename IT, typename Iter>
struct VectorIterator
    : public std::iterator<std::random_access_iterator_tag, IT, uoffset_t> {

  typedef std::iterator<std::random_access_iterator_tag, IT, uoffset_t> super_type;

public:
  VectorIterator(const Iter& data, uoffset_t i) :
      data_(data + IndirectHelper<T>::element_stride * i) {}
  VectorIterator(const VectorIterator &other) : data_(other.data_) {}
  #ifndef FLATBUFFERS_CPP98_STL
  VectorIterator(VectorIterator &&other) : data_(std::move(other.data_)) {}
  #endif

  VectorIterator &operator=(const VectorIterator &other) {
    data_ = other.data_;
    return *this;
  }

  VectorIterator &operator=(VectorIterator &&other) {
    data_ = other.data_;
    return *this;
  }

  bool operator==(const VectorIterator &other) const {
    return data_ == other.data_;
  }

  bool operator!=(const VectorIterator &other) const {
    return data_ != other.data_;
  }

  ptrdiff_t operator-(const VectorIterator &other) const {
    return (data_ - other.data_) / IndirectHelper<T>::element_stride;
  }

  typename super_type::value_type operator *() const {
    return IndirectHelper<T>::Read(data_, 0);
  }

  typename super_type::value_type operator->() const {
    return IndirectHelper<T>::Read(data_, 0);
  }

  VectorIterator &operator++() {
    data_ += IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator++(int) {
    VectorIterator temp(data_, 0);
    data_ += IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator+(const uoffset_t &offset) {
    return VectorIterator(data_ + offset * IndirectHelper<T>::element_stride, 0);
  }

  VectorIterator& operator+=(const uoffset_t &offset) {
    data_ += offset * IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator &operator--() {
    data_ -= IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator--(int) {
    VectorIterator temp(data_, 0);
    data_ -= IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator-(const uoffset_t &offset) {
    return VectorIterator(data_ - offset * IndirectHelper<T>::element_stride, 0);
  }

  VectorIterator& operator-=(const uoffset_t &offset) {
    data_ -= offset * IndirectHelper<T>::element_stride;
    return *this;
  }

private:
  Iter data_;
};

// This is used as a helper type for accessing vectors.
// Vector::data() assumes the vector elements start after the length field.
template<typename T, typename Iter> class Vector {
public:
  typedef VectorIterator<T, typename IndirectHelper<T>::mutable_return_type, Iter>
    iterator;
  typedef VectorIterator<T, typename IndirectHelper<T>::return_type, Iter>
    const_iterator;

  Vector(Iter data):
      data_(data)
  {}

  uoffset_t size() const { return EndianScalar(extractValue<uoffset_t>(data_)); }

  // Deprecated: use size(). Here for backwards compatibility.
  uoffset_t Length() const { return size(); }

  typedef typename IndirectHelper<T>::return_type return_type;
  typedef typename IndirectHelper<T>::mutable_return_type mutable_return_type;

  return_type Get(uoffset_t i) const {
    assert(i < size());
    return IndirectHelper<T>::Read(Data(), i);
  }

  return_type operator[](uoffset_t i) const { return Get(i); }

  // If this is a Vector of enums, T will be its storage type, not the enum
  // type. This function makes it convenient to retrieve value with enum
  // type E.
  template<typename E> E GetEnum(uoffset_t i) const {
    return static_cast<E>(Get(i));
  }

  const Iter GetStructFromOffset(size_t o) const {
    return Data() + o;
  }

  iterator begin() { return iterator(Data(), 0); }
  const_iterator begin() const { return const_iterator(Data(), 0); }

  iterator end() { return iterator(Data(), size()); }
  const_iterator end() const { return const_iterator(Data(), size()); }

  // The raw data in little endian format. Use with care.
  const Iter Data() const {
    return data_ + sizeof(uoffset_t);
  }

  Iter Data() {
    return data_ + sizeof(uoffset_t);
  }

  template<typename K> return_type LookupByKey(K key) const {
    auto search_result = std::lower_bound(begin(), end(), key, KeyCompare<K>);

    if (search_result == end() || (*search_result)->KeyCompareWithValue(key) != 0) {
      return std::nullopt;  // Key not found.
    }

    return *search_result;
  }

  operator Iter() const
  {
      return data_;
  }

protected:
  Iter data_;

private:
  template<typename K> static int KeyCompare(const return_type& ap, const K& bp) {
    return ap->KeyCompareWithValue(bp) < 0;
  }
};

// Represent a vector much like the template above, but in this case we
// don't know what the element types are (used with reflection.h).
template <typename Iter>
class VectorOfAny {
public:
  VectorOfAny(Iter data):
      data_(data)
  {}

  uoffset_t size() const { return EndianScalar(extractValue<uoffset_t>(data_)); }

  const Iter Data() const {
    return data_;
  }
  Iter Data() {
    return data_;
  }
protected:

  Iter data_;
};

// Convenient helper function to get the length of any vector, regardless
// of wether it is null or not (the field is not set).
template<typename T, typename Iter> static inline size_t VectorLength(const std::optional<Vector<T, Iter>> &v) {
  return v ? v->Length() : 0;
}

template <typename Iter> struct String : public Vector<char, Iter> {
  using Vector<char,Iter>::Vector;
  using Vector<char,Iter>::data_;

  std::string str() const {
      if (hasContiguous(data_, sizeof(uoffset_t) + this->Length()))
          return std::string(reinterpret_cast<const char*>(getRawPointer(data_)) + sizeof(uoffset_t), this->Length());
      return std::string(this->begin(), this->begin() + this->Length()); }

  bool operator <(const String &o) const {
    return str() < o.str();
  }
};

// Converts a Field ID to a virtual table offset.
inline voffset_t FieldIndexToOffset(voffset_t field_id) {
  // Should correspond to what EndTable() below builds up.
  const int fixed_fields = 2;  // Vtable size and Object Size.
  return static_cast<voffset_t>((field_id + fixed_fields) * sizeof(voffset_t));
}

/// @endcond

/// @cond FLATBUFFERS_INTERNAL
template<typename T, typename Iter> std::optional<T> GetMutableRoot(Iter begin) {
  flatbuffers::EndianCheck();
  return T(begin + EndianScalar(extractValue<uoffset_t>(begin)));
}

template<typename T, typename Iter> std::optional<T> GetRoot(Iter begin) {
  return GetMutableRoot<T, Iter>(begin);
}

template<typename T, typename Iter> std::optional<T> GetSizePrefixedRoot(Iter buf) {
  return GetRoot<T, Iter>(buf + sizeof(uoffset_t));
}

// Helper to see if the identifier in a buffer has the expected value.

template <typename Iter> inline bool BufferHasIdentifier(const Iter& buf, const char *identifier) {
  return std::equal(
      identifier,
      identifier + std::min(std::strlen(identifier) + 1, static_cast<std::size_t>(FLATBUFFERS_FILE_IDENTIFIER_LENGTH)),
      buf + sizeof(uoffset_t));
}

// Helper class to verify the integrity of a FlatBuffer
template <typename Iter>
class Verifier FLATBUFFERS_FINAL_CLASS {
 public:
  Verifier(const Iter& buf, size_t buf_len, size_t _max_depth = 64,
           size_t _max_tables = 1000000)
    : buf_(buf), end_(buf + buf_len), depth_(0), max_depth_(_max_depth),
      num_tables_(0), max_tables_(_max_tables)
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
        , upper_bound_(buf)
    #endif
    {}

  // Central location where any verification failures register.
  bool Check(bool ok) const {
    #ifdef FLATBUFFERS_DEBUG_VERIFICATION_FAILURE
      assert(ok);
    #endif
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
      if (!ok)
        upper_bound_ = buf_;
    #endif
    return ok;
  }

  // Verify any range within the buffer.
  bool Verify(const Iter& elem, size_t elem_len) const {
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
      auto upper_bound = elem + elem_len;
      if (upper_bound_ < upper_bound)
        upper_bound_ =  upper_bound;
    #endif
    return Check(elem_len <= (size_t) (end_ - buf_) &&
                 elem >= buf_ &&
                 elem <= end_ - elem_len);
  }

  // Verify a range indicated by sizeof(T).
  template<typename T> bool Verify(const Iter& elem) const {
    return Verify(elem, sizeof(T));
  }

  template<typename T> bool VerifyTable(const std::optional<T>& table) {
    return !table || table->Verify(*this);
  }

  template<typename T> bool Verify(const std::optional<Vector<T, Iter>>& vec) const {
    Iter end;
    return !vec ||
           VerifyVector(static_cast<Iter>(*vec), sizeof(T),
                        &end);
  }

  template<typename T> bool Verify(const std::optional<Vector<const T, Iter>>& vec) const {
    return Verify(*reinterpret_cast<const std::optional<Vector<T, Iter>> *>(&vec));
  }

  bool Verify(const std::optional<String<Iter>>& str) const {
    Iter end;
    return !str ||
           (VerifyVector(static_cast<Iter>(*str), 1, &end) &&
            Verify(end, 1) &&      // Must have terminator
            Check(*end == '\0'));  // Terminating byte must be 0.
  }

  // Common code between vectors and strings.
  bool VerifyVector(const Iter& vec, size_t elem_size,
                    Iter *end) const {
    // Check we can read the size field.
    if (!Verify<uoffset_t>(vec)) return false;
    // Check the whole array. If this is a string, the byte past the array
    // must be 0.
    auto size = ReadScalar<uoffset_t>(vec);
    auto max_elems = FLATBUFFERS_MAX_BUFFER_SIZE / elem_size;
    if (!Check(size < max_elems))
      return false;  // Protect against byte_size overflowing.
    auto byte_size = sizeof(size) + elem_size * size;
    *end = vec + byte_size;
    return Verify(vec, byte_size);
  }

  // Special case for string contents, after the above has been called.
  bool VerifyVectorOfStrings(const std::optional<Vector<Offset<String<Iter>>, Iter>>& vec) const {
      if (vec) {
        for (uoffset_t i = 0; i < vec->size(); i++) {
          if (!Verify(vec->Get(i))) return false;
        }
      }
      return true;
  }

  // Special case for table contents, after the above has been called.
  template<typename T> bool VerifyVectorOfTables(const std::optional<Vector<Offset<T>, Iter>>& vec) {
    if (vec) {
      for (uoffset_t i = 0; i < vec->size(); i++) {
        if (!vec->Get(i)->Verify(*this)) return false;
      }
    }
    return true;
  }

  template<typename T> bool VerifyBufferFromStart(const char *identifier,
                                                  const Iter& start) {
    if (identifier &&
        (static_cast<std::size_t>(end_ - start) < 2 * sizeof(flatbuffers_iter::uoffset_t) ||
         !BufferHasIdentifier(start, identifier))) {
      return false;
    }

    // Call T::Verify, which must be in the generated code for this type.
    return Verify<uoffset_t>(start) &&
      T(start + ReadScalar<uoffset_t>(start)).
        Verify(*this)
        #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
          && GetComputedSize()
        #endif
            ;
  }

  // Verify this whole buffer, starting with root type T.
  template<typename T> bool VerifyBuffer(const char *identifier) {
    return VerifyBufferFromStart<T>(identifier, buf_);
  }

  template<typename T> bool VerifySizePrefixedBuffer(const char *identifier) {
    return Verify<uoffset_t>(buf_) &&
           ReadScalar<uoffset_t>(buf_) == end_ - buf_ - sizeof(uoffset_t) &&
           VerifyBufferFromStart<T>(identifier, buf_ + sizeof(uoffset_t));
  }

  // Called at the start of a table to increase counters measuring data
  // structure depth and amount, and possibly bails out with false if
  // limits set by the constructor have been hit. Needs to be balanced
  // with EndTable().
  bool VerifyComplexity() {
    depth_++;
    num_tables_++;
    return Check(depth_ <= max_depth_ && num_tables_ <= max_tables_);
  }

  // Called at the end of a table to pop the depth count.
  bool EndTable() {
    depth_--;
    return true;
  }

  #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
  // Returns the message size in bytes
  size_t GetComputedSize() const {
    uintptr_t size = upper_bound_ - buf_;
    // Align the size to uoffset_t
    size = (size - 1 + sizeof(uoffset_t)) & ~(sizeof(uoffset_t) - 1);
    return (buf_  + size > end_) ?  0 : size;
  }
  #endif

 private:
  const Iter buf_;
  const Iter end_;
  size_t depth_;
  size_t max_depth_;
  size_t num_tables_;
  size_t max_tables_;
#ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
  mutable const Iter upper_bound_;
#endif
};

// "structs" are flat structures that do not have an offset table, thus
// always have all members present and do not support forwards/backwards
// compatible extensions.
template <typename Iter>
class Struct FLATBUFFERS_FINAL_CLASS {
 public:
  template<typename T> T GetField(uoffset_t o) const {
    return ReadScalar<T>(data_ + o);
  }

  template<typename T> T GetStruct(uoffset_t o) const {
    return T(data_ + o);
  }

 private:
  Iter data_;
};

// "tables" use an offset table (possibly shared) that allows fields to be
// omitted and added at will, but uses an extra indirection to read.
template<typename Iter>
class Table {
 public:
  Table(Iter data): data_(data) {}

  const Iter GetVTable() const {
    return data_ - ReadScalar<soffset_t>(data_);
  }

  // This gets the field offset for any of the functions below it, or 0
  // if the field was not present.
  voffset_t GetOptionalFieldOffset(voffset_t field) const {
    // The vtable offset is always at the start.
    auto vtable = GetVTable();
    // The first element is the size of the vtable (fields + type id + itself).
    auto vtsize = ReadScalar<voffset_t>(vtable);
    // If the field we're accessing is outside the vtable, we're reading older
    // data, so it's the same as if the offset was 0 (not present).
    return field < vtsize ? ReadScalar<voffset_t>(vtable + field) : 0;
  }

  template<typename T> T GetField(voffset_t field, T defaultval) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return field_offset ? ReadScalar<T>(data_ + field_offset) : defaultval;
  }

  template<typename P> std::optional<P> GetPointer(voffset_t field) {
    auto field_offset = GetOptionalFieldOffset(field);
    auto p = data_ + field_offset;
    return field_offset ? std::optional<P>(P(p + ReadScalar<uoffset_t>(p))) : std::nullopt;
  }

  template<typename P> std::optional<P> GetPointer(voffset_t field) const {
    return const_cast<Table *>(this)->template GetPointer<P>(field);
  }

  template<typename P> P GetStruct(voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    auto p = data_ + field_offset;
    return extractValue<P>(p);
  }

  bool CheckField(voffset_t field) const {
    return GetOptionalFieldOffset(field) != 0;
  }

  // Verify the vtable of this table.
  // Call this once per table, followed by VerifyField once per field.
  bool VerifyTableStart(Verifier<Iter> &verifier) const {
    // Check the vtable offset.
    if (!verifier.template Verify<soffset_t>(data_)) return false;
    auto vtable = GetVTable();
    // Check the vtable size field, then check vtable fits in its entirety.
    return verifier.VerifyComplexity() &&
           verifier.template Verify<voffset_t>(vtable) &&
           (ReadScalar<voffset_t>(vtable) & (sizeof(voffset_t) - 1)) == 0 &&
           verifier.Verify(vtable, ReadScalar<voffset_t>(vtable));
  }

  // Verify a particular field.
  template<typename T>
  bool VerifyField(const Verifier<Iter> &verifier, voffset_t field,
                   size_t align) const {
    // Calling GetOptionalFieldOffset should be safe now thanks to
    // VerifyTable().
    auto field_offset = GetOptionalFieldOffset(field);
    // Check the actual field.
    return !field_offset || verifier.template VerifyField<T>(data_, field_offset, align);
  }

  // VerifyField for required fields.
  template<typename T>
  bool VerifyFieldRequired(const Verifier<Iter> &verifier, voffset_t field,
                           size_t align) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return verifier.Check(field_offset != 0) &&
           verifier.template VerifyField<T>(data_, field_offset, align);
  }

  // Versions for offsets.
  template<typename T> bool VerifyOffset(const Verifier<Iter> &verifier,
                                        voffset_t field) const {
    // Calling GetOptionalFieldOffset should be safe now thanks to
    // VerifyTable().
    auto field_offset = GetOptionalFieldOffset(field);
    // Check the actual field.
    return !field_offset || verifier.template Verify<T>(data_ + field_offset);
  }

  template<typename T> bool VerifyOffsetRequired(const Verifier<Iter> &verifier,
                                        voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return verifier.Check(field_offset != 0) &&
           verifier.template Verify<T>(data_ + field_offset);
  } 

 private:
  Iter data_;
};
/// @endcond
}  // namespace flatbuffers_iter
}  // namespace maps
}  // namespace yandex

#endif  // FLATBUFFERS_H_
