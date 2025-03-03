/* -----------------------------------------------------------------------------
 * director.swg
 *
 * This file contains support for director classes so that Go proxy
 * methods can be called from C++.
 * ----------------------------------------------------------------------------- */

#include <exception>
#include <map>

namespace Swig {

  class DirectorException : public std::exception {
  };
}

/* Handle memory management for directors.  */

namespace {
  struct GCItem {
    virtual ~GCItem() {}
  };

  struct GCItem_var {
    GCItem_var(GCItem *item = 0) : _item(item) {
    }

    GCItem_var& operator=(GCItem *item) {
      GCItem *tmp = _item;
      _item = item;
      delete tmp;
      return *this;
    }

    ~GCItem_var() {
      delete _item;
    }

    GCItem* operator->() {
      return _item;
    }

    private:
      GCItem *_item;
  };

  template <typename Type>
  struct GCItem_T : GCItem {
    GCItem_T(Type *ptr) : _ptr(ptr) {
    }

    virtual ~GCItem_T() {
      delete _ptr;
    }

  private:
    Type *_ptr;
  };
}

class Swig_memory {
public:
  template <typename Type>
  void swig_acquire_pointer(Type* vptr) {
    if (vptr) {
      swig_owner[vptr] = new GCItem_T<Type>(vptr);
    }
  }
private:
  typedef std::map<void *, GCItem_var> swig_ownership_map;
  swig_ownership_map swig_owner;
};

template <typename Type>
static void swig_acquire_pointer(Swig_memory** pmem, Type* ptr) {
  if (!pmem) {
    *pmem = new Swig_memory;
  }
  (*pmem)->swig_acquire_pointer(ptr);
}
