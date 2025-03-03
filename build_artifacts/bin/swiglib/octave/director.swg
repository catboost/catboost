/* -----------------------------------------------------------------------------
 * director.swg
 *
 * This file contains support for director classes so that Octave proxy
 * methods can be called from C++.
 * ----------------------------------------------------------------------------- */

# define SWIG_DIRECTOR_CAST(ARG) dynamic_cast<Swig::Director *>(ARG)

namespace Swig {

  class Director {
    octave_swig_type *self;
    bool swig_disowned;

    Director(const Director &x);
    Director &operator=(const Director &rhs);
  public:

    Director(void *vptr):self(0), swig_disowned(false) {
      set_rtdir(vptr, this);
    }

    ~Director() {
      swig_director_destroyed(self, this);
      if (swig_disowned)
	self->decref();
    }

    void swig_set_self(octave_swig_type *new_self) {
      assert(!swig_disowned);
      self = new_self;
    }

    octave_swig_type *swig_get_self() const {
      return self;
    }

    void swig_disown() {
      if (swig_disowned)
	return;
      swig_disowned = true;
      self->incref();
    }
  };

  // Base class for director exceptions.
  class DirectorException : public std::exception {
  public:
    static void raise(const char *msg) {
      // ... todo
      throw DirectorException();
    }

    static void raise(const octave_value &ov, const char *msg) {
      // ... todo
      raise(msg);
    }
  };

  class DirectorTypeMismatchException : public DirectorException {
  public:
    static void raise(const char *msg) {
      // ... todo
      throw DirectorTypeMismatchException();
    }

    static void raise(const octave_value &ov, const char *msg) {
      // ... todo
      raise(msg);
    }
  };

  class DirectorPureVirtualException : public DirectorException {
  public:
    static void raise(const char *msg) {
      // ... todo
      throw DirectorPureVirtualException();
    }

    static void raise(const octave_value &ov, const char *msg) {
      // ... todo
      raise(msg);
    }
  };

  SWIGINTERN rtdir_map *get_rtdir_map() {
    static swig_module_info *module = 0;
    if (!module)
      module = SWIG_GetModule(0);
    if (!module)
      return 0;
    if (!module->clientdata)
      module->clientdata = new rtdir_map;
    return (rtdir_map *) module->clientdata;
  }

  SWIGINTERNINLINE void set_rtdir(void *vptr, Director *d) {
    rtdir_map *rm = get_rtdir_map();
    if (rm)
      (*rm)[vptr] = d;
  }

  SWIGINTERNINLINE void erase_rtdir(void *vptr) {
    rtdir_map *rm = get_rtdir_map();
    if (rm)
      (*rm).erase(vptr);
  }

  SWIGINTERNINLINE Director *get_rtdir(void *vptr) {
    rtdir_map *rm = get_rtdir_map();
    if (!rm)
      return 0;
    rtdir_map::const_iterator pos = rm->find(vptr);
    Director *rtdir = (pos != rm->end())? pos->second : 0;
    return rtdir;
  }

  SWIGRUNTIME void swig_director_destroyed(octave_swig_type *self, Director *d) {
    self->director_destroyed(d);
  }

  SWIGRUNTIME octave_swig_type *swig_director_get_self(Director *d) {
    return d->swig_get_self();
  }

  SWIGRUNTIME void swig_director_set_self(Director *d, octave_swig_type *self) {
    d->swig_set_self(self);
  }

}

SWIGRUNTIME void swig_acquire_ownership(void *vptr) {
  //  assert(0);
  // ... todo
}

SWIGRUNTIME void swig_acquire_ownership_array(void *vptr) {
  //  assert(0);
  // ... todo
}

SWIGRUNTIME void swig_acquire_ownership_obj(void *vptr, int own) {
  //  assert(0);
  // ... todo
}
