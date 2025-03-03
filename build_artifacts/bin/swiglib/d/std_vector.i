/* -----------------------------------------------------------------------------
 * std_vector.i
 *
 * SWIG typemaps for std::vector<T>, D implementation.
 *
 * The D wrapper is made to loosely resemble a tango.util.container.more.Vector
 * and to provide built-in array-like access.
 *
 * If T does define an operator==, then use the SWIG_STD_VECTOR_ENHANCED
 * macro to obtain enhanced functionality (none yet), for example:
 *
 *   SWIG_STD_VECTOR_ENHANCED(SomeNamespace::Klass)
 *   %template(VectKlass) std::vector<SomeNamespace::Klass>;
 *
 * Warning: heavy macro usage in this file. Use swig -E to get a sane view on
 * the real file contents!
 * ----------------------------------------------------------------------------- */

// Warning: Use the typemaps here in the expectation that the macros they are in will change name.

%include <std_common.i>

// MACRO for use within the std::vector class body
%define SWIG_STD_VECTOR_MINIMUM_INTERNAL(CONST_REFERENCE, CTYPE...)
#if (SWIG_D_VERSION == 1)
%typemap(dimports) std::vector< CTYPE > "static import tango.core.Exception;"
%proxycode %{
public this($typemap(dtype, CTYPE)[] values) {
  this();
  append(values);
}

alias push_back add;
alias push_back push;
alias push_back opCatAssign;
alias size length;
alias opSlice slice;

public $typemap(dtype, CTYPE) opIndexAssign($typemap(dtype, CTYPE) value, size_t index) {
  if (index >= size()) {
    throw new tango.core.Exception.NoSuchElementException("Tried to assign to element out of vector bounds.");
  }
  setElement(index, value);
  return value;
}

public $typemap(dtype, CTYPE) opIndex(size_t index) {
  if (index >= size()) {
    throw new tango.core.Exception.NoSuchElementException("Tried to read from element out of vector bounds.");
  }
  return getElement(index);
}

public void append($typemap(dtype, CTYPE)[] value...) {
  foreach (v; value) {
    add(v);
  }
}

public $typemap(dtype, CTYPE)[] opSlice() {
  $typemap(dtype, CTYPE)[] array = new $typemap(dtype, CTYPE)[size()];
  foreach (i, ref value; array) {
    value = getElement(i);
  }
  return array;
}

public int opApply(int delegate(ref $typemap(dtype, CTYPE) value) dg) {
  int result;

  size_t currentSize = size();
  for (size_t i = 0; i < currentSize; ++i) {
    auto value = getElement(i);
    result = dg(value);
    setElement(i, value);
  }
  return result;
}

public int opApply(int delegate(ref size_t index, ref $typemap(dtype, CTYPE) value) dg) {
  int result;

  size_t currentSize = size();
  for (size_t i = 0; i < currentSize; ++i) {
    auto value = getElement(i);

    // Workaround for http://d.puremagic.com/issues/show_bug.cgi?id=2443.
    auto index = i;

    result = dg(index, value);
    setElement(i, value);
  }
  return result;
}

public void capacity(size_t value) {
  if (value < size()) {
    throw new tango.core.Exception.IllegalArgumentException("Tried to make the capacity of a vector smaller than its size.");
  }

  reserve(value);
}
%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef CTYPE value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef CONST_REFERENCE const_reference;

    void clear();
    void push_back(CTYPE const& x);
    size_type size() const;
    size_type capacity() const;
    void reserve(size_type n) throw (std::length_error);

    vector();
    vector(const vector &other);

    %extend {
      vector(size_type capacity) throw (std::length_error) {
        std::vector< CTYPE >* pv = 0;
        pv = new std::vector< CTYPE >();

        // Might throw std::length_error.
        pv->reserve(capacity);

        return pv;
      }

      size_type unused() const {
        return $self->capacity() - $self->size();
      }

      const_reference remove() throw (std::out_of_range) {
        if ($self->empty()) {
          throw std::out_of_range("Tried to remove last element from empty vector.");
        }

        std::vector< CTYPE >::const_reference value = $self->back();
        $self->pop_back();
        return value;
      }

      const_reference remove(size_type index) throw (std::out_of_range) {
        if (index >= $self->size()) {
          throw std::out_of_range("Tried to remove element with invalid index.");
        }

        std::vector< CTYPE >::iterator it = $self->begin() + index;
        std::vector< CTYPE >::const_reference value = *it;
        $self->erase(it);
        return value;
      }
    }

    // Wrappers for setting/getting items with the possibly thrown exception
    // specified (important for SWIG wrapper generation).
    %extend {
      const_reference getElement(size_type index) throw (std::out_of_range) {
        if ((index < 0) || ($self->size() <= index)) {
          throw std::out_of_range("Tried to get value of element with invalid index.");
        }
        return (*$self)[index];
      }
    }

    // Use CTYPE const& instead of const_reference to work around SWIG code
    // generation issue when using const pointers as vector elements (like
    // std::vector< const int* >).
    %extend {
      void setElement(size_type index, CTYPE const& val) throw (std::out_of_range) {
        if ((index < 0) || ($self->size() <= index)) {
          throw std::out_of_range("Tried to set value of element with invalid index.");
        }
        (*$self)[index] = val;
      }
    }

%dmethodmodifiers std::vector::getElement "private"
%dmethodmodifiers std::vector::setElement "private"
%dmethodmodifiers std::vector::reserve "private"

#else

%typemap(dimports) std::vector< CTYPE > %{
static import std.algorithm;
static import std.exception;
static import std.range;
static import std.traits;
%}
%proxycode %{
alias size_t KeyType;
alias $typemap(dtype, CTYPE) ValueType;

this(ValueType[] values...) {
  this();
  reserve(values.length);
  foreach (e; values) {
    this ~= e;
  }
}

struct Range {
  private $typemap(dtype, std::vector< CTYPE >) _outer;
  private size_t _a, _b;

  this($typemap(dtype, std::vector< CTYPE >) data, size_t a, size_t b) {
    _outer = data;
    _a = a;
    _b = b;
  }

  @property bool empty() const {
    assert((cast($typemap(dtype, std::vector< CTYPE >))_outer).length >= _b);
    return _a >= _b;
  }

  @property Range save() {
    return this;
  }

  @property ValueType front() {
    std.exception.enforce(!empty);
    return _outer[_a];
  }

  @property void front(ValueType value) {
    std.exception.enforce(!empty);
    _outer[_a] = std.algorithm.move(value);
  }

  void popFront() {
    std.exception.enforce(!empty);
    ++_a;
  }

  void opIndexAssign(ValueType value, size_t i) {
    i += _a;
    std.exception.enforce(i < _b && _b <= _outer.length);
    _outer[i] = value;
  }

  void opIndexOpAssign(string op)(ValueType value, size_t i) {
    std.exception.enforce(_outer && _a + i < _b && _b <= _outer.length);
    auto element = _outer[i];
    mixin("element "~op~"= value;");
    _outer[i] = element;
  }
}

// TODO: dup?

Range opSlice() {
  return Range(this, 0, length);
}

Range opSlice(size_t a, size_t b) {
  std.exception.enforce(a <= b && b <= length);
  return Range(this, a, b);
}

size_t opDollar() const {
  return length;
}

@property ValueType front() {
  std.exception.enforce(!empty);
  return getElement(0);
}

@property void front(ValueType value) {
  std.exception.enforce(!empty);
  setElement(0, value);
}

@property ValueType back() {
  std.exception.enforce(!empty);
  return getElement(length - 1);
}

@property void back(ValueType value) {
  std.exception.enforce(!empty);
  setElement(length - 1, value);
}

ValueType opIndex(size_t i) {
  return getElement(i);
}

void opIndexAssign(ValueType value, size_t i) {
  setElement(i, value);
}

void opIndexOpAssign(string op)(ValueType value, size_t i) {
  auto element = this[i];
  mixin("element "~op~"= value;");
  this[i] = element;
}

ValueType[] opBinary(string op, Stuff)(Stuff stuff) if (op == "~") {
  ValueType[] result;
  result ~= this[];
  assert(result.length == length);
  result ~= stuff[];
  return result;
}

void opOpAssign(string op, Stuff)(Stuff stuff) if (op == "~") {
  static if (is(typeof(insertBack(stuff)))) {
    insertBack(stuff);
  } else if (is(typeof(insertBack(stuff[])))) {
    insertBack(stuff[]);
  } else {
    static assert(false, "Cannot append " ~ Stuff.stringof ~ " to " ~ typeof(this).stringof);
  }
}

alias size length;

alias remove removeAny;
alias removeAny stableRemoveAny;

size_t insertBack(Stuff)(Stuff stuff)
if (std.traits.isImplicitlyConvertible!(Stuff, ValueType)){
  push_back(stuff);
  return 1;
}
size_t insertBack(Stuff)(Stuff stuff)
if (std.range.isInputRange!Stuff &&
    std.traits.isImplicitlyConvertible!(std.range.ElementType!Stuff, ValueType)) {
  size_t itemCount;
  foreach(item; stuff) {
    insertBack(item);
    ++itemCount;
  }
  return itemCount;
}
alias insertBack insert;

alias pop_back removeBack;
alias pop_back stableRemoveBack;

size_t insertBefore(Stuff)(Range r, Stuff stuff)
if (std.traits.isImplicitlyConvertible!(Stuff, ValueType)) {
  std.exception.enforce(r._outer.swigCPtr == swigCPtr && r._a < length);
  insertAt(r._a, stuff);
  return 1;
}

size_t insertBefore(Stuff)(Range r, Stuff stuff)
if (std.range.isInputRange!Stuff && std.traits.isImplicitlyConvertible!(ElementType!Stuff, ValueType)) {
  std.exception.enforce(r._outer.swigCPtr == swigCPtr && r._a <= length);

  size_t insertCount;
  foreach(i, item; stuff) {
    insertAt(r._a + i, item);
    ++insertCount;
  }

  return insertCount;
}

size_t insertAfter(Stuff)(Range r, Stuff stuff) {
  // TODO: optimize
  immutable offset = r._a + r.length;
  std.exception.enforce(offset <= length);
  auto result = insertBack(stuff);
  std.algorithm.bringToFront(this[offset .. length - result],
    this[length - result .. length]);
  return result;
}

size_t replace(Stuff)(Range r, Stuff stuff)
if (std.range.isInputRange!Stuff &&
    std.traits.isImplicitlyConvertible!(ElementType!Stuff, ValueType)) {
  immutable offset = r._a;
  std.exception.enforce(offset <= length);
  size_t result;
  for (; !stuff.empty; stuff.popFront()) {
    if (r.empty) {
      // append the rest
      return result + insertBack(stuff);
    }
    r.front = stuff.front;
    r.popFront();
    ++result;
  }
  // Remove remaining stuff in r
  remove(r);
  return result;
}

size_t replace(Stuff)(Range r, Stuff stuff)
if (std.traits.isImplicitlyConvertible!(Stuff, ValueType))
{
    if (r.empty)
    {
        insertBefore(r, stuff);
    }
    else
    {
        r.front = stuff;
        r.popFront();
        remove(r);
    }
    return 1;
}

Range linearRemove(Range r) {
  std.exception.enforce(r._a <= r._b && r._b <= length);
  immutable tailLength = length - r._b;
  linearRemove(r._a, r._b);
  return this[length - tailLength .. length];
}
alias remove stableLinearRemove;

int opApply(int delegate(ref $typemap(dtype, CTYPE) value) dg) {
  int result;

  size_t currentSize = size();
  for (size_t i = 0; i < currentSize; ++i) {
    auto value = getElement(i);
    result = dg(value);
    setElement(i, value);
  }
  return result;
}

int opApply(int delegate(ref size_t index, ref $typemap(dtype, CTYPE) value) dg) {
  int result;

  size_t currentSize = size();
  for (size_t i = 0; i < currentSize; ++i) {
    auto value = getElement(i);

    // Workaround for http://d.puremagic.com/issues/show_bug.cgi?id=2443.
    auto index = i;

    result = dg(index, value);
    setElement(i, value);
  }
  return result;
}
%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef CTYPE value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef CONST_REFERENCE const_reference;

    bool empty() const;
    void clear();
    void push_back(CTYPE const& x);
    void pop_back();
    size_type size() const;
    size_type capacity() const;
    void reserve(size_type n) throw (std::length_error);

    vector();
    vector(const vector &other);

    %extend {
      vector(size_type capacity) throw (std::length_error) {
        std::vector< CTYPE >* pv = 0;
        pv = new std::vector< CTYPE >();

        // Might throw std::length_error.
        pv->reserve(capacity);

        return pv;
      }

      const_reference remove() throw (std::out_of_range) {
        if ($self->empty()) {
          throw std::out_of_range("Tried to remove last element from empty vector.");
        }

        std::vector< CTYPE >::const_reference value = $self->back();
        $self->pop_back();
        return value;
      }

      const_reference remove(size_type index) throw (std::out_of_range) {
        if (index >= $self->size()) {
          throw std::out_of_range("Tried to remove element with invalid index.");
        }

        std::vector< CTYPE >::iterator it = $self->begin() + index;
        std::vector< CTYPE >::const_reference value = *it;
        $self->erase(it);
        return value;
      }

      void removeBack(size_type how_many) throw (std::out_of_range) {
        std::vector< CTYPE >::iterator end = $self->end();
        std::vector< CTYPE >::iterator start = end - how_many;
        $self->erase(start, end);
      }

      void linearRemove(size_type start_index, size_type end_index) throw (std::out_of_range) {
        std::vector< CTYPE >::iterator start = $self->begin() + start_index;
        std::vector< CTYPE >::iterator end = $self->begin() + end_index;
        $self->erase(start, end);
      }

      void insertAt(size_type index, CTYPE const& x) throw (std::out_of_range) {
        std::vector< CTYPE >::iterator it = $self->begin() + index;
        $self->insert(it, x);
      }
    }

    // Wrappers for setting/getting items with the possibly thrown exception
    // specified (important for SWIG wrapper generation).
    %extend {
      const_reference getElement(size_type index) throw (std::out_of_range) {
        if ((index < 0) || ($self->size() <= index)) {
          throw std::out_of_range("Tried to get value of element with invalid index.");
        }
        return (*$self)[index];
      }
    }
    // Use CTYPE const& instead of const_reference to work around SWIG code
    // generation issue when using const pointers as vector elements (like
    // std::vector< const int* >).
    %extend {
      void setElement(size_type index, CTYPE const& val) throw (std::out_of_range) {
        if ((index < 0) || ($self->size() <= index)) {
          throw std::out_of_range("Tried to set value of element with invalid index.");
        }
        (*$self)[index] = val;
      }
    }

%dmethodmodifiers std::vector::getElement "private"
%dmethodmodifiers std::vector::setElement "private"
#endif
%enddef

// Extra methods added to the collection class if operator== is defined for the class being wrapped
// The class will then implement IList<>, which adds extra functionality
%define SWIG_STD_VECTOR_EXTRA_OP_EQUALS_EQUALS(CTYPE...)
    %extend {
    }
%enddef

// For vararg handling in macros, from swigmacros.swg
#define %arg(X...) X

// Macros for std::vector class specializations/enhancements
%define SWIG_STD_VECTOR_ENHANCED(CTYPE...)
namespace std {
  template<> class vector<CTYPE > {
    SWIG_STD_VECTOR_MINIMUM_INTERNAL(const value_type&, %arg(CTYPE))
    SWIG_STD_VECTOR_EXTRA_OP_EQUALS_EQUALS(CTYPE)
  };
}
%enddef

%{
#include <vector>
#include <stdexcept>
%}

namespace std {
  // primary (unspecialized) class template for std::vector
  // does not require operator== to be defined
  template<class T> class vector {
    SWIG_STD_VECTOR_MINIMUM_INTERNAL(const value_type&, T)
  };
  // specializations for pointers
  template<class T> class vector<T *> {
    SWIG_STD_VECTOR_MINIMUM_INTERNAL(const value_type&, T *)
    SWIG_STD_VECTOR_EXTRA_OP_EQUALS_EQUALS(T *)
  };
  // bool is a bit different in the C++ standard - const_reference in particular
  template<> class vector<bool> {
    SWIG_STD_VECTOR_MINIMUM_INTERNAL(bool, bool)
    SWIG_STD_VECTOR_EXTRA_OP_EQUALS_EQUALS(bool)
  };
}

// template specializations for std::vector
// these provide extra collections methods as operator== is defined
SWIG_STD_VECTOR_ENHANCED(char)
SWIG_STD_VECTOR_ENHANCED(signed char)
SWIG_STD_VECTOR_ENHANCED(unsigned char)
SWIG_STD_VECTOR_ENHANCED(short)
SWIG_STD_VECTOR_ENHANCED(unsigned short)
SWIG_STD_VECTOR_ENHANCED(int)
SWIG_STD_VECTOR_ENHANCED(unsigned int)
SWIG_STD_VECTOR_ENHANCED(long)
SWIG_STD_VECTOR_ENHANCED(unsigned long)
SWIG_STD_VECTOR_ENHANCED(long long)
SWIG_STD_VECTOR_ENHANCED(unsigned long long)
SWIG_STD_VECTOR_ENHANCED(float)
SWIG_STD_VECTOR_ENHANCED(double)
SWIG_STD_VECTOR_ENHANCED(std::string) // also requires a %include <std_string.i>
