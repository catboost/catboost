/* -----------------------------------------------------------------------------
 * std_list.i
 *
 * SWIG typemaps for std::list<T>
 * C# implementation
 * The C# wrapper is made to look and feel like a C# System.Collections.Generic.LinkedList<> collection.
 *
 * Note that IEnumerable<> is implemented in the proxy class which is useful for using LINQ with
 * C++ std::list wrappers. The ICollection<> interface is also implemented to provide enhanced functionality
 * whenever we are confident that the required C++ operator== is available. This is the case for when
 * T is a primitive type or a pointer. If T does define an operator==, then use the SWIG_STD_LIST_ENHANCED
 * macro to obtain this enhanced functionality, for example:
 *
 *   SWIG_STD_LIST_ENHANCED(SomeNamespace::Klass)
 *   %template(ListKlass) std::list<SomeNamespace::Klass>;
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// MACRO for use within the std::list class body
%define SWIG_STD_LIST_MINIMUM_INTERNAL(CSINTERFACE, CTYPE...)
%typemap(csinterfaces) std::list< CTYPE > "global::System.IDisposable, global::System.Collections.IEnumerable, global::System.Collections.Generic.CSINTERFACE<$typemap(cstype, CTYPE)>\n";

%apply void *VOID_INT_PTR { std::list< CTYPE >::iterator * };

%proxycode %{
  public $csclassname(global::System.Collections.IEnumerable c) : this() {
    if (c == null)
      throw new global::System.ArgumentNullException("c");
    foreach ($typemap(cstype, CTYPE) element in c) {
      this.AddLast(element);
    }
  }

  public bool IsReadOnly {
    get {
      return false;
    }
  }

  public int Count {
    get {
      return (int)size();
    }
  }

  public $csclassnameNode First {
    get {
      if (Count == 0)
        return null;
      return new $csclassnameNode(getFirstIter(), this);
    }
  }

  public $csclassnameNode Last {
    get {
      if (Count == 0)
        return null;
      return new $csclassnameNode(getLastIter(), this);
    }
  }

  public $csclassnameNode AddFirst($typemap(cstype, CTYPE) value) {
    push_front(value);
    return new $csclassnameNode(getFirstIter(), this);
  }

  public void AddFirst($csclassnameNode newNode) {
    ValidateNewNode(newNode);
    if (!newNode.inlist) {
      push_front(newNode.csharpvalue);
      newNode.iter = getFirstIter();
      newNode.inlist = true;
    } else {
      throw new global::System.InvalidOperationException("The " + newNode.GetType().Name + " node already belongs to a " + this.GetType().Name);
    }
  }

  public $csclassnameNode AddLast($typemap(cstype, CTYPE) value) {
    push_back(value);
    return new $csclassnameNode(getLastIter(), this);
  }

  public void AddLast($csclassnameNode newNode) {
    ValidateNewNode(newNode);
    if (!newNode.inlist) {
      push_back(newNode.csharpvalue);
      newNode.iter = getLastIter();
      newNode.inlist = true;
    } else {
      throw new global::System.InvalidOperationException("The " + newNode.GetType().Name + " node already belongs to a " + this.GetType().Name);
    }
  }

  public $csclassnameNode AddBefore($csclassnameNode node, $typemap(cstype, CTYPE) value) {
    return new $csclassnameNode(insertNode(node.iter, value), this);
  }

  public void AddBefore($csclassnameNode node, $csclassnameNode newNode) {
    ValidateNode(node);
    ValidateNewNode(newNode);
    if (!newNode.inlist) {
      newNode.iter = insertNode(node.iter, newNode.csharpvalue);
      newNode.inlist = true;
    } else {
      throw new global::System.InvalidOperationException("The " + newNode.GetType().Name + " node already belongs to a " + this.GetType().Name);
    }
  }

  public $csclassnameNode AddAfter($csclassnameNode node, $typemap(cstype, CTYPE) value) {
    node = node.Next;
    return new $csclassnameNode(insertNode(node.iter, value), this);
  }

  public void AddAfter($csclassnameNode node, $csclassnameNode newNode) {
    ValidateNode(node);
    ValidateNewNode(newNode);
    if (!newNode.inlist) {
      if (node == this.Last)
        AddLast(newNode);
      else
      {
        node = node.Next;
        newNode.iter = insertNode(node.iter, newNode.csharpvalue);
        newNode.inlist = true;
      }
    } else {
      throw new global::System.InvalidOperationException("The " + newNode.GetType().Name + " node already belongs to a " + this.GetType().Name);
    }
  }

  public void Add($typemap(cstype, CTYPE) value) {
    AddLast(value);
  }

  public void Remove($csclassnameNode node) {
    ValidateNode(node);
    eraseIter(node.iter);
  }

  public void CopyTo($typemap(cstype, CTYPE)[] array, int index) {
    if (array == null)
      throw new global::System.ArgumentNullException("array");
    if (index < 0 || index > array.Length)
      throw new global::System.ArgumentOutOfRangeException("index", "Value is less than zero");
    if (array.Rank > 1)
      throw new global::System.ArgumentException("Multi dimensional array.", "array");
    $csclassnameNode node = this.First;
    if (node != null) {
      do {
        array[index++] = node.Value;
        node = node.Next;
      } while (node != null);
    }
  }

  internal void ValidateNode($csclassnameNode node) {
    if (node == null) {
      throw new System.ArgumentNullException("node");
    }
    if (!node.inlist || node.list != this) {
      throw new System.InvalidOperationException("node");
    }
  }

  internal void ValidateNewNode($csclassnameNode node) {
    if (node == null) {
      throw new System.ArgumentNullException("node");
    }
  }

  global::System.Collections.Generic.IEnumerator<$typemap(cstype, CTYPE)> global::System.Collections.Generic.IEnumerable<$typemap(cstype, CTYPE)>.GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  global::System.Collections.IEnumerator global::System.Collections.IEnumerable.GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  public $csclassnameEnumerator GetEnumerator() {
    return new $csclassnameEnumerator(this);
  }

  public sealed class $csclassnameEnumerator : global::System.Collections.IEnumerator,
    global::System.Collections.Generic.IEnumerator<$typemap(cstype, CTYPE)>
  {
    private $csclassname collectionRef;
    private $csclassnameNode currentNode;
    private int currentIndex;
    private object currentObject;
    private int currentSize;

    public $csclassnameEnumerator($csclassname collection) {
      collectionRef = collection;
      currentNode = collection.First;
      currentIndex = 0;
      currentObject = null;
      currentSize = collectionRef.Count;
    }

    // Type-safe iterator Current
    public $typemap(cstype, CTYPE) Current {
      get {
        if (currentIndex == -1)
          throw new global::System.InvalidOperationException("Enumeration not started.");
        if (currentIndex > currentSize)
          throw new global::System.InvalidOperationException("Enumeration finished.");
        if (currentObject == null)
          throw new global::System.InvalidOperationException("Collection modified.");
        return ($typemap(cstype, CTYPE))currentObject;
      }
    }

    // Type-unsafe IEnumerator.Current
    object global::System.Collections.IEnumerator.Current {
      get {
        return Current;
      }
    }

    public bool MoveNext() {
      if (currentNode == null) {
        currentIndex = collectionRef.Count + 1;
        return false;
      }
      ++currentIndex;
      currentObject = currentNode.Value;
      currentNode = currentNode.Next;
      return true;
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

  public sealed class $csclassnameNode {
    internal $csclassname list;
    internal System.IntPtr iter;
    internal $typemap(cstype, CTYPE) csharpvalue;
    internal bool inlist;

    public $csclassnameNode($typemap(cstype, CTYPE) value) {
      csharpvalue = value;
      inlist = false;
    }

    internal $csclassnameNode(System.IntPtr iter, $csclassname list) {
      this.list = list;
      this.iter = iter;
      inlist = true;
    }

    public $csclassname List {
      get {
        return this.list;
      }
    }

    public $csclassnameNode Next {
      get {
        if (list.getNextIter(iter) == System.IntPtr.Zero)
          return null;
        return new $csclassnameNode(list.getNextIter(iter), list);
      }
    }

    public $csclassnameNode Previous {
      get {
        if (list.getPrevIter(iter) == System.IntPtr.Zero)
          return null;
        return new $csclassnameNode(list.getPrevIter(iter), list);
      }
    }

    public $typemap(cstype, CTYPE) Value {
      get {
        return list.getItem(this.iter);
      }
      set {
        list.setItem(this.iter, value);
      }
    }

    public static bool operator==($csclassnameNode node1, $csclassnameNode node2) {
      if (object.ReferenceEquals(node1, null) && object.ReferenceEquals(node2, null))
        return true;
      if (object.ReferenceEquals(node1, null) || object.ReferenceEquals(node2, null))
        return false;
      return node1.Equals(node2);
    }

    public static bool operator!=($csclassnameNode node1, $csclassnameNode node2) {
      if (node1 == null && node2 == null)
        return false;
      if (node1 == null || node2 == null)
        return true;
      return !node1.Equals(node2);
    }

    public bool Equals($csclassnameNode node) {
      if (node == null)
        return false;
      if (!node.inlist || !this.inlist)
        return object.ReferenceEquals(this, node);
      return list.equals(this.iter, node.iter);
    }

    public override bool Equals(object node) {
      return Equals(($csclassnameNode)node);
    }

    public override int GetHashCode() {
      int hash = 13;
      if (inlist) {
        hash = (hash * 7) + this.list.GetHashCode();
        hash = (hash * 7) + this.Value.GetHashCode();
        hash = (hash * 7) + this.list.getNextIter(this.iter).GetHashCode();
        hash = (hash * 7) + this.list.getPrevIter(this.iter).GetHashCode();
      } else {
        hash = (hash * 7) + this.csharpvalue.GetHashCode();
      }
      return hash;
    }

    public void Dispose() {
      list.deleteIter(this.iter);
    }
  }
%}

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef CTYPE value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  class iterator;

  void push_front(CTYPE const& x);
  void push_back(CTYPE const& x);
  %rename(RemoveFirst) pop_front;
  void pop_front();
  %rename(RemoveLast) pop_back;
  void pop_back();
  size_type size() const;
  %rename(Clear) clear;
  void clear();
  %extend {
    const_reference getItem(iterator *iter) {
      return **iter;
    }

    void setItem(iterator *iter, CTYPE const& val) {
      *(*iter) = val;
    }

    iterator *getFirstIter() {
      if ($self->size() == 0)
        return NULL;
      return new std::list< CTYPE >::iterator($self->begin());
    }

    iterator *getLastIter() {
      if ($self->size() == 0)
        return NULL;
      return new std::list< CTYPE >::iterator(--$self->end());
    }

    iterator *getNextIter(iterator *iter) {
      std::list< CTYPE >::iterator it = *iter;
      if (std::distance(it, --$self->end()) != 0) {
        std::list< CTYPE >::iterator* itnext = new std::list< CTYPE >::iterator(++it);
        return itnext;
      }
      return NULL;
    }

    iterator *getPrevIter(iterator *iter) {
      std::list< CTYPE >::iterator it = *iter;
      if (std::distance($self->begin(), it) != 0) {
        std::list< CTYPE >::iterator* itprev = new std::list< CTYPE >::iterator(--it);
        return itprev;
      }
      return NULL;
    }

    iterator *insertNode(iterator *iter, CTYPE const& value) {
      std::list< CTYPE >::iterator it = $self->insert(*iter, value);
      return new std::list< CTYPE >::iterator(it);
    }

    void eraseIter(iterator *iter) {
      std::list< CTYPE >::iterator it = *iter;
      $self->erase(it);
    }

    void deleteIter(iterator *iter) {
      delete iter;
    }

    bool equals(iterator *iter1, iterator *iter2) {
      if (iter1 == NULL && iter2 == NULL)
        return true;
      std::list< CTYPE >::iterator it1 = *iter1;
      std::list< CTYPE >::iterator it2 = *iter2;
      return it1 == it2;
    }
  }
%enddef

// Extra methods added to the collection class if operator== is defined for the class being wrapped
// The class will then implement ICollection<>, which adds extra functionality
%define SWIG_STD_LIST_EXTRA_OP_EQUALS_EQUALS(CTYPE...)
  %extend {
    bool Contains(CTYPE const& value) {
      return std::find($self->begin(), $self->end(), value) != $self->end();
    }

    bool Remove(CTYPE const& value) {
      std::list< CTYPE >::iterator it = std::find($self->begin(), $self->end(), value);
      if (it != $self->end()) {
        $self->erase(it);
        return true;
      }
      return false;
    }

    iterator *find(CTYPE const& value) {
      if (std::find($self->begin(), $self->end(), value) != $self->end()) {
        return new std::list< CTYPE >::iterator(std::find($self->begin(), $self->end(), value));
      }
      return NULL;
    }
  }
%proxycode %{
  public $csclassnameNode Find($typemap(cstype, CTYPE) value) {
    System.IntPtr tmp = find(value);
    if (tmp != System.IntPtr.Zero) {
      return new $csclassnameNode(tmp, this);
    }
    return null;
  }
%}
%enddef

// Macros for std::list class specializations/enhancements
%define SWIG_STD_LIST_ENHANCED(CTYPE...)
namespace std {
  template<> class list< CTYPE > {
    SWIG_STD_LIST_MINIMUM_INTERNAL(ICollection, %arg(CTYPE));
    SWIG_STD_LIST_EXTRA_OP_EQUALS_EQUALS(CTYPE)
  };
}
%enddef


%{
#include <list>
#include <algorithm>
#include <stdexcept>
%}

%csmethodmodifiers std::list::size "private"
%csmethodmodifiers std::list::getItem "private"
%csmethodmodifiers std::list::setItem "private"
%csmethodmodifiers std::list::push_front "private"
%csmethodmodifiers std::list::push_back "private"
%csmethodmodifiers std::list::getFirstIter "private"
%csmethodmodifiers std::list::getNextIter "private"
%csmethodmodifiers std::list::getPrevIter "private"
%csmethodmodifiers std::list::getLastIter "private"
%csmethodmodifiers std::list::find "private"
%csmethodmodifiers std::list::deleteIter "private"

namespace std {
  // primary (unspecialized) class template for std::list
  // does not require operator== to be defined
  template<class T>
  class list {
    SWIG_STD_LIST_MINIMUM_INTERNAL(IEnumerable, T)
  };
  // specialization for pointers
  template<class T>
  class list<T *> {
    SWIG_STD_LIST_MINIMUM_INTERNAL(ICollection, T *)
    SWIG_STD_LIST_EXTRA_OP_EQUALS_EQUALS(T *)
  };
}

// template specializations for std::list
// these provide extra collections methods as operator== is defined
SWIG_STD_LIST_ENHANCED(char)
SWIG_STD_LIST_ENHANCED(signed char)
SWIG_STD_LIST_ENHANCED(unsigned char)
SWIG_STD_LIST_ENHANCED(short)
SWIG_STD_LIST_ENHANCED(unsigned short)
SWIG_STD_LIST_ENHANCED(int)
SWIG_STD_LIST_ENHANCED(unsigned int)
SWIG_STD_LIST_ENHANCED(long)
SWIG_STD_LIST_ENHANCED(unsigned long)
SWIG_STD_LIST_ENHANCED(long long)
SWIG_STD_LIST_ENHANCED(unsigned long long)
SWIG_STD_LIST_ENHANCED(float)
SWIG_STD_LIST_ENHANCED(double)
SWIG_STD_LIST_ENHANCED(std::string) // also requires a %include <std_string.i>
SWIG_STD_LIST_ENHANCED(std::wstring) // also requires a %include <std_wstring.i>
