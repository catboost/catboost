/* -----------------------------------------------------------------------------
 * std_unordered_set.i
 *
 * SWIG typemaps for std::unordered_set
 * The Java proxy class extends java.util.AbstractSet. The std::unordered_set
 * container looks and feels much like a java.util.HashSet from Java.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::unordered_set
// ------------------------------------------------------------------------

%{
#include <unordered_set>
#include <stdexcept>
%}

%fragment("SWIG_UnorderedSetSize", "header", fragment="SWIG_JavaIntFromSize_t") {
  SWIGINTERN jint SWIG_UnorderedSetSize(size_t size) {
    jint sz = SWIG_JavaIntFromSize_t(size);
    if (sz == -1) {
      throw std::out_of_range("unordered_set size is too large to fit into a Java int");
    }

    return sz;
  }
}

%javamethodmodifiers std::unordered_set::sizeImpl "private";
%javamethodmodifiers std::unordered_set::containsImpl "private";
%javamethodmodifiers std::unordered_set::removeImpl "private";
%javamethodmodifiers std::unordered_set::hasNextImpl "private";
%javamethodmodifiers std::unordered_set::begin "private";
%javamethodmodifiers std::unordered_set::end "private";

%rename(Iterator) std::unordered_set::iterator;
%nodefaultctor std::unordered_set::iterator;
%javamethodmodifiers std::unordered_set::iterator::incrementUnchecked "private";
%javamethodmodifiers std::unordered_set::iterator::derefUnchecked "private";
%javamethodmodifiers std::unordered_set::iterator::isNot "private";

namespace std {

template <class Key>
class unordered_set {

%typemap(javabase) std::unordered_set<Key> "java.util.AbstractSet<$typemap(jboxtype, Key)>"
%proxycode %{
  public $javaclassname(java.util.Collection<? extends $typemap(jboxtype, Key)> collection) {
    this();
    addAll(collection);
  }

  public int size() {
    return sizeImpl();
  }

  public boolean addAll(java.util.Collection<? extends $typemap(jboxtype, Key)> collection) {
    boolean didAddElement = false;
    for (java.lang.Object object : collection) {
      didAddElement |= add(($typemap(jboxtype, Key))object);
    }

    return didAddElement;
  }

  public java.util.Iterator<$typemap(jboxtype, Key)> iterator() {
    return new java.util.Iterator<$typemap(jboxtype, Key)>() {
      private Iterator curr;
      private Iterator end;

      private java.util.Iterator<$typemap(jboxtype, Key)> init() {
        curr = $javaclassname.this.begin();
        end = $javaclassname.this.end();
        return this;
      }

      public $typemap(jboxtype, Key) next() {
        if (!hasNext()) {
          throw new java.util.NoSuchElementException();
        }

        // Save the current position, increment it,
        // then return the value at the position before the increment.
        final $typemap(jboxtype, Key) currValue = curr.derefUnchecked();
        curr.incrementUnchecked();
        return currValue;
      }

      public boolean hasNext() {
        return curr.isNot(end);
      }
    }.init();
  }

  public boolean containsAll(java.util.Collection<?> collection) {
    for (java.lang.Object object : collection) {
      if (!contains(object)) {
        return false;
      }
    }

    return true;
  }

  public boolean contains(java.lang.Object object) {
    if (!(object instanceof $typemap(jboxtype, Key))) {
      return false;
    }

    return containsImpl(($typemap(jboxtype, Key))object);
  }

  public boolean removeAll(java.util.Collection<?> collection) {
    boolean didRemoveElement = false;
    for (java.lang.Object object : collection) {
      didRemoveElement |= remove(object);
    }

    return didRemoveElement;
  }

  public boolean remove(java.lang.Object object) {
    if (!(object instanceof $typemap(jboxtype, Key))) {
      return false;
    }

    return removeImpl(($typemap(jboxtype, Key))object);
  }
%}

  public:

    struct iterator {
      %typemap(javaclassmodifiers) iterator "protected class"
      %extend {
        void incrementUnchecked() {
          ++(*$self);
        }

        Key derefUnchecked() const {
          return **$self;
        }

        bool isNot(iterator other) const {
          return (*$self != other);
        }
      }
    };

    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef Key value_type;
    typedef Key key_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    unordered_set();
    unordered_set(const unordered_set& other);

    %rename(isEmpty) empty;
    bool empty() const;
    void clear();
    iterator begin();
    iterator end();

    %extend {
      %fragment("SWIG_UnorderedSetSize");

      // Returns whether item was inserted.
      bool add(const Key& key) {
        return self->insert(key).second;
      }

      // Returns whether set contains key.
      bool containsImpl(const Key& key) {
        return (self->count(key) > 0);
      }

      // Returns whether the item was erased.
      bool removeImpl(const Key& key) {
        return (self->erase(key) > 0);
      }

      jint sizeImpl() const throw (std::out_of_range) {
        return SWIG_UnorderedSetSize(self->size());
      }

      bool hasNextImpl(const iterator& itr) const {
        return (itr != $self->end());
      }
    }
};

}
