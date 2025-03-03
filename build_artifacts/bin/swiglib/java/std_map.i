/* -----------------------------------------------------------------------------
 * std_map.i
 *
 * SWIG typemaps for std::map
 * The Java proxy class extends java.util.AbstractMap. The std::map
 * container looks and feels much like a java.util.HashMap from Java.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::map
// ------------------------------------------------------------------------

%{
#include <map>
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

%javamethodmodifiers std::map::sizeImpl "private";
%javamethodmodifiers std::map::containsImpl "private";
%javamethodmodifiers std::map::putUnchecked "private";
%javamethodmodifiers std::map::removeUnchecked "private";
%javamethodmodifiers std::map::find "private";
%javamethodmodifiers std::map::begin "private";
%javamethodmodifiers std::map::end "private";

%rename(Iterator) std::map::iterator;
%nodefaultctor std::map::iterator;
%javamethodmodifiers std::map::iterator::getNextUnchecked "private";
%javamethodmodifiers std::map::iterator::isNot "private";
%javamethodmodifiers std::map::iterator::getKey "private";
%javamethodmodifiers std::map::iterator::getValue "private";
%javamethodmodifiers std::map::iterator::setValue "private";

namespace std {

template<class K, class T, class C = std::less< K> > class map {

%typemap(javabase) std::map< K, T, C >
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

    map();
    map(const map& other);

    struct iterator {
      %typemap(javaclassmodifiers) iterator "protected class"
      %extend {
        std::map< K, T, C >::iterator getNextUnchecked() {
          std::map< K, T, C >::iterator copy = (*$self);
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

      void removeUnchecked(const std::map< K, T, C >::iterator itr) {
        self->erase(itr);
      }
    }
};

// Legacy macros (deprecated)
%define specialize_std_map_on_key(K,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_key ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_value(T,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_value ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_both(K,CHECK_K,CONVERT_K_FROM,CONVERT_K_TO, T,CHECK_T,CONVERT_T_FROM,CONVERT_T_TO)
#warning "specialize_std_map_on_both ignored - macro is deprecated and no longer necessary"
%enddef

}
