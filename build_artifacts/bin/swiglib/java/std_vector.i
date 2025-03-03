/* -----------------------------------------------------------------------------
 * std_vector.i
 *
 * SWIG typemaps for std::vector.
 * The Java proxy class extends java.util.AbstractList and implements
 * java.util.RandomAccess. The std::vector container looks and feels much like a
 * java.util.ArrayList from Java.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

%{
#include <vector>
#include <stdexcept>
%}

%fragment("SWIG_VectorSize", "header", fragment="SWIG_JavaIntFromSize_t") {
SWIGINTERN jint SWIG_VectorSize(size_t size) {
  jint sz = SWIG_JavaIntFromSize_t(size);
  if (sz == -1)
    throw std::out_of_range("vector size is too large to fit into a Java int");
  return sz;
}
}

%define SWIG_STD_VECTOR_MINIMUM_INTERNAL(CTYPE, CONST_REFERENCE)
%typemap(javabase) std::vector< CTYPE > "java.util.AbstractList<$typemap(jboxtype, CTYPE)>"
%typemap(javainterfaces) std::vector< CTYPE > "java.util.RandomAccess"
%proxycode %{
  public $javaclassname($typemap(jstype, CTYPE)[] initialElements) {
    this();
    reserve(initialElements.length);

    for ($typemap(jstype, CTYPE) element : initialElements) {
      add(element);
    }
  }

  public $javaclassname(Iterable<$typemap(jboxtype, CTYPE)> initialElements) {
    this();
    for ($typemap(jstype, CTYPE) element : initialElements) {
      add(element);
    }
  }

  public $typemap(jboxtype, CTYPE) get(int index) {
    return doGet(index);
  }

  public $typemap(jboxtype, CTYPE) set(int index, $typemap(jboxtype, CTYPE) e) {
    return doSet(index, e);
  }

  public boolean add($typemap(jboxtype, CTYPE) e) {
    modCount++;
    doAdd(e);
    return true;
  }

  public void add(int index, $typemap(jboxtype, CTYPE) e) {
    modCount++;
    doAdd(index, e);
  }

  public $typemap(jboxtype, CTYPE) remove(int index) {
    modCount++;
    return doRemove(index);
  }

  protected void removeRange(int fromIndex, int toIndex) {
    modCount++;
    doRemoveRange(fromIndex, toIndex);
  }

  public int size() {
    return doSize();
  }
%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef CTYPE value_type;
    typedef CTYPE *pointer;
    typedef CTYPE const *const_pointer;
    typedef CTYPE &reference;
    typedef CONST_REFERENCE const_reference;

    vector();
    vector(const vector &other);

    size_type capacity() const;
    void reserve(size_type n) throw (std::length_error);
    %rename(isEmpty) empty;
    bool empty() const;
    void clear();
    %extend {
      %fragment("SWIG_VectorSize");

      vector(jint count, const CTYPE &value) throw (std::out_of_range) {
        if (count < 0)
          throw std::out_of_range("vector count must be positive");
        return new std::vector< CTYPE >(static_cast<std::vector< CTYPE >::size_type>(count), value);
      }

      jint doSize() const throw (std::out_of_range) {
        return SWIG_VectorSize(self->size());
      }

      void doAdd(const value_type& x) {
        self->push_back(x);
      }

      void doAdd(jint index, const value_type& x) throw (std::out_of_range) {
        jint size = static_cast<jint>(self->size());
        if (0 <= index && index <= size) {
          self->insert(self->begin() + index, x);
        } else {
          throw std::out_of_range("vector index out of range");
        }
      }

      value_type doRemove(jint index) throw (std::out_of_range) {
        jint size = static_cast<jint>(self->size());
        if (0 <= index && index < size) {
          CTYPE const old_value = (*self)[index];
          self->erase(self->begin() + index);
          return old_value;
        } else {
          throw std::out_of_range("vector index out of range");
        }
      }

      CONST_REFERENCE doGet(jint index) throw (std::out_of_range) {
        jint size = static_cast<jint>(self->size());
        if (index >= 0 && index < size)
          return (*self)[index];
        else
          throw std::out_of_range("vector index out of range");
      }

      value_type doSet(jint index, const value_type& val) throw (std::out_of_range) {
        jint size = static_cast<jint>(self->size());
        if (index >= 0 && index < size) {
          CTYPE const old_value = (*self)[index];
          (*self)[index] = val;
          return old_value;
        }
        else
          throw std::out_of_range("vector index out of range");
      }

      void doRemoveRange(jint fromIndex, jint toIndex) throw (std::out_of_range) {
        jint size = static_cast<jint>(self->size());
        if (0 <= fromIndex && fromIndex <= toIndex && toIndex <= size) {
          self->erase(self->begin() + fromIndex, self->begin() + toIndex);
        } else {
          throw std::out_of_range("vector index out of range");
        }
      }
    }
%enddef

%javamethodmodifiers std::vector::doSize        "private";
%javamethodmodifiers std::vector::doAdd         "private";
%javamethodmodifiers std::vector::doGet         "private";
%javamethodmodifiers std::vector::doSet         "private";
%javamethodmodifiers std::vector::doRemove      "private";
%javamethodmodifiers std::vector::doRemoveRange "private";

namespace std {

    template<class T> class vector {
        SWIG_STD_VECTOR_MINIMUM_INTERNAL(T, const value_type&)
    };

    // bool specialization
    template<> class vector<bool> {
        SWIG_STD_VECTOR_MINIMUM_INTERNAL(bool, bool)
    };
}

%define specialize_std_vector(T)
#warning "specialize_std_vector - specialization for type T no longer needed"
%enddef
