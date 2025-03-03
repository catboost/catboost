/* -----------------------------------------------------------------------------
 * std_list.i
 *
 * SWIG typemaps for std::list.
 * The Java proxy class extends java.util.AbstractSequentialList. The std::list
 * container looks and feels much like a java.util.LinkedList from Java.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

%{
#include <list>
#include <stdexcept>
%}

%fragment("SWIG_ListSize", "header", fragment="SWIG_JavaIntFromSize_t") {
SWIGINTERN jint SWIG_ListSize(size_t size) {
  jint sz = SWIG_JavaIntFromSize_t(size);
  if (sz == -1)
    throw std::out_of_range("list size is too large to fit into a Java int");
  return sz;
}
}

%javamethodmodifiers std::list::begin "private";
%javamethodmodifiers std::list::insert "private";
%javamethodmodifiers std::list::doSize "private";
%javamethodmodifiers std::list::doPreviousIndex "private";
%javamethodmodifiers std::list::doNextIndex "private";
%javamethodmodifiers std::list::doHasNext "private";

// Match Java style better:
%rename(Iterator) std::list::iterator;

%nodefaultctor std::list::iterator;

namespace std {
  template <typename T> class list {

%typemap(javabase) std::list<T> "java.util.AbstractSequentialList<$typemap(jboxtype, T)>"
%proxycode %{
  public $javaclassname(java.util.Collection c) {
    this();
    java.util.ListIterator<$typemap(jboxtype, T)> it = listIterator(0);
    // Special case the "copy constructor" here to avoid lots of cross-language calls
    for (java.lang.Object o : c) {
      it.add(($typemap(jboxtype, T))o);
    }
  }

  public int size() {
    return doSize();
  }

  public boolean add($typemap(jboxtype, T) value) {
    addLast(value);
    return true;
  }

  public java.util.ListIterator<$typemap(jboxtype, T)> listIterator(int index) {
    return new java.util.ListIterator<$typemap(jboxtype, T)>() {
      private Iterator pos;
      private Iterator last;

      private java.util.ListIterator<$typemap(jboxtype, T)> init(int index) {
        if (index < 0 || index > $javaclassname.this.size())
          throw new IndexOutOfBoundsException("Index: " + index);
        pos = $javaclassname.this.begin();
	pos = pos.advance_unchecked(index);
        return this;
      }

      public void add($typemap(jboxtype, T) v) {
        // Technically we can invalidate last here, but this makes more sense
        last = $javaclassname.this.insert(pos, v);
      }

      public void set($typemap(jboxtype, T) v) {
        if (null == last) {
          throw new IllegalStateException();
        }
        last.set_unchecked(v);
      }

      public void remove() {
        if (null == last) {
          throw new IllegalStateException();
        }
        $javaclassname.this.remove(last);
        last = null;
      }

      public int previousIndex() {
        return $javaclassname.this.doPreviousIndex(pos);
      }

      public int nextIndex() {
        return $javaclassname.this.doNextIndex(pos);
      }

      public $typemap(jboxtype, T) previous() {
        if (previousIndex() < 0) {
          throw new java.util.NoSuchElementException();
        }
        last = pos;
        pos = pos.previous_unchecked();
        return last.deref_unchecked();
      }

      public $typemap(jboxtype, T) next() {
        if (!hasNext()) {
          throw new java.util.NoSuchElementException();
        }
        last = pos;
        pos = pos.next_unchecked();
        return last.deref_unchecked();
      }

      public boolean hasPrevious() {
        // This call to previousIndex() will be much slower than the hasNext() implementation, but it's simpler like this with C++ forward iterators
        return previousIndex() != -1;
      }

      public boolean hasNext() {
        return $javaclassname.this.doHasNext(pos);
      }
    }.init(index);
  }
%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    /*
     * We'd actually be better off having the nested class *not* be static in the wrapper
     * output, but this doesn't actually remove the $static from the nested class still.
     * (This would allow us to somewhat simplify the implementation of the ListIterator
     * interface and give "natural" semantics to Java users of the C++ iterator)
     */
    //%typemap(javaclassmodifiers) iterator "public class"
    //%typemap(javainterfaces) iterator "java.util.ListIterator<$typemap(jboxtype, T)>"

    struct iterator {
      %extend {
	void set_unchecked(const T &v) {
	  **$self = v;
	}

	iterator next_unchecked() const {
	  std::list<T>::iterator ret = *$self;
	  ++ret;
	  return ret;
	}

	iterator previous_unchecked() const {
	  std::list<T>::iterator ret = *$self;
	  --ret;
	  return ret;
	}

	T deref_unchecked() const {
	  return **$self;
	}

	iterator advance_unchecked(size_type index) const {
	  std::list<T>::iterator ret = *$self;
	  std::advance(ret, index);
	  return ret;
	}
      }
    };

    list();
    list(const list& other);

    %rename(isEmpty) empty;
    bool empty() const;
    void clear();
    %rename(remove) erase;
    iterator erase(iterator pos);
    %rename(removeLast) pop_back;
    void pop_back();
    %rename(removeFirst) pop_front;
    void pop_front();
    %rename(addLast) push_back;
    void push_back(const T &value);
    %rename(addFirst) push_front;
    void push_front(const T &value);
    iterator begin();
    iterator end();
    iterator insert(iterator pos, const T &value);

    %extend {
      %fragment("SWIG_ListSize");

      list(jint count, const T &value) throw (std::out_of_range) {
        if (count < 0)
          throw std::out_of_range("list count must be positive");
        return new std::list<T>(static_cast<std::list<T>::size_type>(count), value);
      }

      jint doSize() const throw (std::out_of_range) {
        return SWIG_ListSize(self->size());
      }

      jint doPreviousIndex(const iterator &pos) const throw (std::out_of_range) {
        return pos == self->begin() ? -1 : SWIG_ListSize(std::distance(self->begin(), static_cast<std::list<T>::const_iterator>(pos)));
      }

      jint doNextIndex(const iterator &pos) const throw (std::out_of_range) {
        return pos == self->end() ? SWIG_ListSize(self->size()) : SWIG_ListSize(std::distance(self->begin(), static_cast<std::list<T>::const_iterator>(pos)));
      }

      bool doHasNext(const iterator &pos) const {
        return pos != $self->end();
      }
    }
  };
}
