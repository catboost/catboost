/* -----------------------------------------------------------------------------
 * std_unordered_map.i
 *
 * SWIG typemaps for std::unordered_map
 * The Java proxy class extends java.util.AbstractMap. The std::unordered_map
 * container looks and feels much like a java.util.HashMap from Java.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::unordered_map
// ------------------------------------------------------------------------

%{
#include <unordered_map>
#include <stdexcept>
%}

%fragment("SWIG_MapSize", "header", fragment="SWIG_JavaIntFromSize_t") {
  SWIGINTERN jint SWIG_MapSize(size_t size) {
    jint sz = SWIG_JavaIntFromSize_t(size);
    if (sz == -1) {
      throw std::out_of_range("map size is too large to fit into a Java int");
    }

    return sz;
  }
}

%javamethodmodifiers std::unordered_map::sizeImpl "private";
%javamethodmodifiers std::unordered_map::containsImpl "private";
%javamethodmodifiers std::unordered_map::putUnchecked "private";
%javamethodmodifiers std::unordered_map::removeUnchecked "private";
%javamethodmodifiers std::unordered_map::find "private";
%javamethodmodifiers std::unordered_map::begin "private";
%javamethodmodifiers std::unordered_map::end "private";

%rename(Iterator) std::unordered_map::iterator;
%nodefaultctor std::unordered_map::iterator;
%javamethodmodifiers std::unordered_map::iterator::getNextUnchecked "private";
%javamethodmodifiers std::unordered_map::iterator::isNot "private";
%javamethodmodifiers std::unordered_map::iterator::getKey "private";
%javamethodmodifiers std::unordered_map::iterator::getValue "private";
%javamethodmodifiers std::unordered_map::iterator::setValue "private";

namespace std {

template<class K, class T> class unordered_map {

%typemap(javabase) std::unordered_map<K, T>
    "java.util.AbstractMap<$typemap(jboxtype, K), $typemap(jboxtype, T)>"

%proxycode %{

  public int size() {
    return sizeImpl();
  }

  public boolean containsKey(java.lang.Object key) {
    if (!(key instanceof $typemap(jboxtype, K))) {
      return false;
    }

    return containsImpl(($typemap(jboxtype, K))key);
  }

  public $typemap(jboxtype, T) get(java.lang.Object key) {
    if (!(key instanceof $typemap(jboxtype, K))) {
      return null;
    }

    Iterator itr = find(($typemap(jboxtype, K)) key);
    if (itr.isNot(end())) {
      return itr.getValue();
    }

    return null;
  }

  public $typemap(jboxtype, T) put($typemap(jboxtype, K) key, $typemap(jboxtype, T) value) {
    Iterator itr = find(($typemap(jboxtype, K)) key);
    if (itr.isNot(end())) {
      $typemap(jboxtype, T) oldValue = itr.getValue();
      itr.setValue(value);
      return oldValue;
    } else {
      putUnchecked(key, value);
      return null;
    }
  }

  public $typemap(jboxtype, T) remove(java.lang.Object key) {
    if (!(key instanceof $typemap(jboxtype, K))) {
      return null;
    }

    Iterator itr = find(($typemap(jboxtype, K)) key);
    if (itr.isNot(end())) {
      $typemap(jboxtype, T) oldValue = itr.getValue();
      removeUnchecked(itr);
      return oldValue;
    } else {
      return null;
    }
  }

  public java.util.Set<Entry<$typemap(jboxtype, K), $typemap(jboxtype, T)>> entrySet() {
    java.util.Set<Entry<$typemap(jboxtype, K), $typemap(jboxtype, T)>> setToReturn =
        new java.util.HashSet<Entry<$typemap(jboxtype, K), $typemap(jboxtype, T)>>();

    Iterator itr = begin();
    final Iterator end = end();
    while (itr.isNot(end)) {
      setToReturn.add(new Entry<$typemap(jboxtype, K), $typemap(jboxtype, T)>() {
        private Iterator iterator;

        private Entry<$typemap(jboxtype, K), $typemap(jboxtype, T)> init(Iterator iterator) {
          this.iterator = iterator;
          return this;
        }

        public $typemap(jboxtype, K) getKey() {
          return iterator.getKey();
        }

        public $typemap(jboxtype, T) getValue() {
          return iterator.getValue();
        }

        public $typemap(jboxtype, T) setValue($typemap(jboxtype, T) newValue) {
          $typemap(jboxtype, T) oldValue = iterator.getValue();
          iterator.setValue(newValue);
          return oldValue;
        }
      }.init(itr));
      itr = itr.getNextUnchecked();
    }

    return setToReturn;
  }
%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef K key_type;
    typedef T mapped_type;
    typedef std::pair< const K, T > value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    unordered_map();
    unordered_map(const unordered_map& other);

    struct iterator {
      %typemap(javaclassmodifiers) iterator "protected class"
      %extend {
        std::unordered_map< K, T >::iterator getNextUnchecked() {
          std::unordered_map< K, T >::iterator copy = (*$self);
          return ++copy;
        }

        bool isNot(iterator other) const {
          return (*$self != other);
        }

        K getKey() const {
          return (*$self)->first;
        }

        T getValue() const {
          return (*$self)->second;
        }

        void setValue(const T& newValue) {
          (*$self)->second = newValue;
        }
      }
    };

    %rename(isEmpty) empty;
    bool empty() const;
    void clear();
    iterator find(const K& key);
    iterator begin();
    iterator end();
    %extend {
      %fragment("SWIG_MapSize");

      jint sizeImpl() const throw (std::out_of_range) {
        return SWIG_MapSize(self->size());
      }

      bool containsImpl(const K& key) {
        return (self->count(key) > 0);
      }

      void putUnchecked(const K& key, const T& value) {
        (*self)[key] = value;
      }

      void removeUnchecked(const std::unordered_map< K, T >::iterator itr) {
        self->erase(itr);
      }
    }
};

}
