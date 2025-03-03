/* -----------------------------------------------------------------------------
 * std_set.i
 *
 * SWIG typemaps for std::set<T>.
 *
 * Note that ISet<> used here requires .NET 4 or later.
 *
 * The C# wrapper implements ISet<> interface and shares performance
 * characteristics of C# System.Collections.Generic.SortedSet<> class, but
 * doesn't provide quite all of its methods.
 * ----------------------------------------------------------------------------- */

%{
#include <set>
#include <algorithm>
#include <stdexcept>
%}

%csmethodmodifiers std::set::size "private"
%csmethodmodifiers std::set::getitem "private"
%csmethodmodifiers std::set::create_iterator_begin "private"
%csmethodmodifiers std::set::get_next "private"
%csmethodmodifiers std::set::destroy_iterator "private"

namespace std {

// TODO: Add support for comparator and allocator template parameters.
template <class T>
class set {

%typemap(csinterfaces) std::set<T> "global::System.IDisposable, global::System.Collections.Generic.ISet<$typemap(cstype, T)>\n";
%proxycode %{
  void global::System.Collections.Generic.ICollection<$typemap(cstype, T)>.Add($typemap(cstype, T) item) {
      ((global::System.Collections.Generic.ISet<$typemap(cstype, T)>)this).Add(item);
  }

  public bool TryGetValue($typemap(cstype, T) equalValue, out $typemap(cstype, T) actualValue) {
    try {
      actualValue = getitem(equalValue);
      return true;
    } catch {
      actualValue = default($typemap(cstype, T));
      return false;
    }
  }

  public int Count {
    get {
      return (int)size();
    }
  }

  public bool IsReadOnly {
    get {
      return false;
    }
  }

  public void CopyTo($typemap(cstype, T)[] array) {
    CopyTo(array, 0);
  }

  public void CopyTo($typemap(cstype, T)[] array, int arrayIndex) {
    if (array == null)
      throw new global::System.ArgumentNullException("array");
    if (arrayIndex < 0)
      throw new global::System.ArgumentOutOfRangeException("arrayIndex", "Value is less than zero");
    if (array.Rank > 1)
      throw new global::System.ArgumentException("Multi dimensional array.", "array");
    if (arrayIndex+this.Count > array.Length)
      throw new global::System.ArgumentException("Number of elements to copy is too large.");

    foreach ($typemap(cstype, T) item in this) {
      array.SetValue(item, arrayIndex++);
    }
  }

  public void ExceptWith(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    foreach ($typemap(cstype, T) item in other) {
      Remove(item);
    }
  }

  public void IntersectWith(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    $csclassname old = new $csclassname(this);

    Clear();
    foreach ($typemap(cstype, T) item in other) {
      if (old.Contains(item))
        Add(item);
    }
  }

  private static int count_enum(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    int count = 0;
    foreach ($typemap(cstype, T) item in other) {
      count++;
    }

    return count;
  }

  public bool IsProperSubsetOf(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    return IsSubsetOf(other) && Count < count_enum(other);
  }

  public bool IsProperSupersetOf(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    return IsSupersetOf(other) && Count > count_enum(other);
  }

  public bool IsSubsetOf(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    int countContained = 0;

    foreach ($typemap(cstype, T) item in other) {
      if (Contains(item))
        countContained++;
    }

    return countContained == Count;
  }

  public bool IsSupersetOf(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    foreach ($typemap(cstype, T) item in other) {
      if (!Contains(item))
        return false;
    }

    return true;
  }

  public bool Overlaps(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    foreach ($typemap(cstype, T) item in other) {
      if (Contains(item))
        return true;
    }

    return false;
  }

  public bool SetEquals(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    return IsSupersetOf(other) && Count == count_enum(other);
  }

  public void SymmetricExceptWith(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    foreach ($typemap(cstype, T) item in other) {
      if (!Remove(item))
        Add(item);
    }
  }

  public void UnionWith(global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)> other) {
    foreach ($typemap(cstype, T) item in other) {
      Add(item);
    }
  }

  private global::System.Collections.Generic.ICollection<$typemap(cstype, T)> Items {
    get {
      global::System.Collections.Generic.ICollection<$typemap(cstype, T)> items = new global::System.Collections.Generic.List<$typemap(cstype, T)>();
      int size = this.Count;
      if (size > 0) {
        global::System.IntPtr iter = create_iterator_begin();
        for (int i = 0; i < size; i++) {
          items.Add(get_next(iter));
        }
        destroy_iterator(iter);
      }
      return items;
    }
  }

  global::System.Collections.Generic.IEnumerator<$typemap(cstype, T)> global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)>.GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  global::System.Collections.IEnumerator global::System.Collections.IEnumerable.GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  public $csclassnameEnumerator GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  // Type-safe enumerator
  /// Note that the IEnumerator documentation requires an InvalidOperationException to be thrown
  /// whenever the collection is modified. This has been done for changes in the size of the
  /// collection but not when one of the elements of the collection is modified as it is a bit
  /// tricky to detect unmanaged code that modifies the collection under our feet.
  public sealed class $csclassnameEnumerator : global::System.Collections.IEnumerator,
      global::System.Collections.Generic.IEnumerator<$typemap(cstype, T)>
  {
    private $csclassname collectionRef;
    private global::System.Collections.Generic.IList<$typemap(cstype, T)> ItemsCollection;
    private int currentIndex;
    private object currentObject;
    private int currentSize;

    public $csclassnameEnumerator($csclassname collection) {
      collectionRef = collection;
      ItemsCollection = new global::System.Collections.Generic.List<$typemap(cstype, T)>(collection.Items);
      currentIndex = -1;
      currentObject = null;
      currentSize = collectionRef.Count;
    }

    // Type-safe iterator Current
    public $typemap(cstype, T) Current {
      get {
        if (currentIndex == -1)
          throw new global::System.InvalidOperationException("Enumeration not started.");
        if (currentIndex > currentSize - 1)
          throw new global::System.InvalidOperationException("Enumeration finished.");
        if (currentObject == null)
          throw new global::System.InvalidOperationException("Collection modified.");
        return ($typemap(cstype, T))currentObject;
      }
    }

    // Type-unsafe IEnumerator.Current
    object global::System.Collections.IEnumerator.Current {
      get {
        return Current;
      }
    }

    public bool MoveNext() {
      int size = collectionRef.Count;
      bool moveOkay = (currentIndex+1 < size) && (size == currentSize);
      if (moveOkay) {
        currentIndex++;
        currentObject = ItemsCollection[currentIndex];
      } else {
        currentObject = null;
      }
      return moveOkay;
    }

    public void Reset() {
      currentIndex = -1;
      currentObject = null;
      if (collectionRef.Count != currentSize) {
        throw new global::System.InvalidOperationException("Collection modified.");
      }
    }

    public void Dispose() {
      currentIndex = -1;
      currentObject = null;
    }
  }

%}

  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T key_type;
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    set();
    set(const set& other);
    size_type size() const;
    bool empty() const;
    %rename(Clear) clear;
    void clear();
    %extend {
      bool Add(const value_type& item) {
        return $self->insert(item).second;
      }

      bool Contains(const value_type& item) {
        return $self->count(item) != 0;
      }

      bool Remove(const value_type& item) {
        return $self->erase(item) != 0;
      }

      const value_type& getitem(const value_type& item) throw (std::out_of_range) {
        std::set<T>::iterator iter = $self->find(item);
        if (iter == $self->end())
          throw std::out_of_range("item not found");

        return *iter;
      }

      // create_iterator_begin(), get_next() and destroy_iterator work together to provide a collection of items to C#
      %apply void *VOID_INT_PTR { std::set<T>::iterator *create_iterator_begin }
      %apply void *VOID_INT_PTR { std::set<T>::iterator *swigiterator }

      std::set<T>::iterator *create_iterator_begin() {
        return new std::set<T>::iterator($self->begin());
      }

      const key_type& get_next(std::set<T>::iterator *swigiterator) {
        std::set<T>::iterator iter = *swigiterator;
        (*swigiterator)++;
        return *iter;
      }

      void destroy_iterator(std::set<T>::iterator *swigiterator) {
        delete swigiterator;
      }
    }
};

}
