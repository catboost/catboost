/* -----------------------------------------------------------------------------
 * std_array.i
 *
 * SWIG typemaps for std::array<T, N>
 * C# implementation
 * The C# wrapper is made to look and feel like a C# System.Collections.Generic.IReadOnlyList<> collection.
 * ----------------------------------------------------------------------------- */

%{
#include <algorithm>
#include <array>
#include <stdexcept>
%}

%include <std_common.i>


%define SWIG_STD_ARRAY_INTERNAL(T, N)
%typemap(csinterfaces) std::array< T, N > "global::System.IDisposable, global::System.Collections.IEnumerable\n    , global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)>\n";
%proxycode %{
  public $csclassname(global::System.Collections.ICollection c) : this() {
    if (c == null)
      throw new global::System.ArgumentNullException("c");
    int end = global::System.Math.Min(this.Count, c.Count);
    int i = 0;
    foreach ($typemap(cstype, T) elem in c) {
      if (i >= end)
        break;
      this[i++] = elem;
    }
  }

  public int Count {
    get {
      return (int)size();
    }
  }

  public $typemap(cstype, T) this[int index]  {
    get {
      return getitem(index);
    }
    set {
      setitem(index, value);
    }
  }

  public bool IsEmpty {
    get {
      return empty();
    }
  }

  public void CopyTo($typemap(cstype, T)[] array)
  {
    CopyTo(0, array, 0, this.Count);
  }

  public void CopyTo($typemap(cstype, T)[] array, int arrayIndex)
  {
    CopyTo(0, array, arrayIndex, this.Count);
  }

  public void CopyTo(int index, $typemap(cstype, T)[] array, int arrayIndex, int count)
  {
    if (array == null)
      throw new global::System.ArgumentNullException("array");
    if (index < 0)
      throw new global::System.ArgumentOutOfRangeException("index", "Value is less than zero");
    if (arrayIndex < 0)
      throw new global::System.ArgumentOutOfRangeException("arrayIndex", "Value is less than zero");
    if (count < 0)
      throw new global::System.ArgumentOutOfRangeException("count", "Value is less than zero");
    if (array.Rank > 1)
      throw new global::System.ArgumentException("Multi dimensional array.", "array");
    if (index+count > this.Count || arrayIndex+count > array.Length)
      throw new global::System.ArgumentException("Number of elements to copy is too large.");
    for (int i=0; i<count; i++)
      array.SetValue(getitemcopy(index+i), arrayIndex+i);
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
  public sealed class $csclassnameEnumerator : global::System.Collections.IEnumerator
    , global::System.Collections.Generic.IEnumerator<$typemap(cstype, T)>
  {
    private $csclassname collectionRef;
    private int currentIndex;
    private object currentObject;
    private int currentSize;

    public $csclassnameEnumerator($csclassname collection) {
      collectionRef = collection;
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
        currentObject = collectionRef[currentIndex];
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
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    array();
    array(const array &other);

    size_type size() const;
    bool empty() const;

    %rename(Fill) fill;
    void fill(const value_type& value);

    %rename(Swap) swap;
    void swap(array& other);

    %extend {
      T getitemcopy(int index) throw (std::out_of_range) {
        if (index>=0 && index<(int)$self->size())
          return (*$self)[index];
        else
          throw std::out_of_range("index");
      }
      const_reference getitem(int index) throw (std::out_of_range) {
        if (index>=0 && index<(int)$self->size())
          return (*$self)[index];
        else
          throw std::out_of_range("index");
      }
      void setitem(int index, const_reference val) throw (std::out_of_range) {
        if (index>=0 && index<(int)$self->size())
          (*$self)[index] = val;
        else
          throw std::out_of_range("index");
      }
      void Reverse() {
        std::reverse($self->begin(), $self->end());
      }
      void Reverse(int index, int count) throw (std::out_of_range, std::invalid_argument) {
        if (index < 0)
          throw std::out_of_range("index");
        if (count < 0)
          throw std::out_of_range("count");
        if (index >= (int)$self->size()+1 || index+count > (int)$self->size())
          throw std::invalid_argument("invalid range");
        std::reverse($self->begin()+index, $self->begin()+index+count);
      }
    }
%enddef


%csmethodmodifiers std::array::empty "private"
%csmethodmodifiers std::array::getitemcopy "private"
%csmethodmodifiers std::array::getitem "private"
%csmethodmodifiers std::array::setitem "private"
%csmethodmodifiers std::array::size "private"

namespace std {
  template<class T, size_t N> class array {
    SWIG_STD_ARRAY_INTERNAL(T, N)
  };
}
