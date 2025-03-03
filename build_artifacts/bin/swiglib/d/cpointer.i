/* -----------------------------------------------------------------------------
 * cpointer.i
 *
 * D-specific version of ../cpointer.i.
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * %pointer_class(type,name)
 *
 * Places a simple proxy around a simple type like 'int', 'float', or whatever.
 * The proxy provides this interface:
 *
 *       class type {
 *       public:
 *           type();
 *          ~type();
 *           type value();
 *           void assign(type value);
 *       };
 *
 * Example:
 *
 *    %pointer_class(int, intp);
 *
 *    int add(int *x, int *y) { return *x + *y; }
 *
 * In python (with proxies)
 *
 *    >>> a = intp()
 *    >>> a.assign(10)
 *    >>> a.value()
 *    10
 *    >>> b = intp()
 *    >>> b.assign(20)
 *    >>> print add(a,b)
 *    30
 *
 * As a general rule, this macro should not be used on class/structures that
 * are already defined in the interface.
 * ----------------------------------------------------------------------------- */


%define %pointer_class(TYPE, NAME)
%{
typedef TYPE NAME;
%}

typedef struct {
} NAME;

%extend NAME {
#ifdef __cplusplus
NAME() {
  return new TYPE();
}
~NAME() {
  if (self) delete self;
}
#else
NAME() {
  return (TYPE *) calloc(1,sizeof(TYPE));
}
~NAME() {
  if (self) free(self);
}
#endif
}

%extend NAME {

void assign(TYPE value) {
  *self = value;
}
TYPE value() {
  return *self;
}
TYPE * ptr() {
  return self;
}
static NAME * frompointer(TYPE *t) {
  return (NAME *) t;
}

}

%types(NAME = TYPE);

%enddef

/* -----------------------------------------------------------------------------
 * %pointer_functions(type,name)
 *
 * Create functions for allocating/deallocating pointers.   This can be used
 * if you don't want to create a proxy class or if the pointer is complex.
 *
 *    %pointer_functions(int, intp)
 *
 *    int add(int *x, int *y) { return *x + *y; }
 *
 * In python (with proxies)
 *
 *    >>> a = copy_intp(10)
 *    >>> intp_value(a)
 *    10
 *    >>> b = new_intp()
 *    >>> intp_assign(b,20)
 *    >>> print add(a,b)
 *    30
 *    >>> delete_intp(a)
 *    >>> delete_intp(b)
 *
 * ----------------------------------------------------------------------------- */

%define %pointer_functions(TYPE,NAME)
%{
static TYPE *new_##NAME() { %}
#ifdef __cplusplus
%{  return new TYPE(); %}
#else
%{  return (TYPE *) calloc(1,sizeof(TYPE)); %}
#endif
%{}

static TYPE *copy_##NAME(TYPE value) { %}
#ifdef __cplusplus
%{  return new TYPE(value); %}
#else
%{  TYPE *self = (TYPE *) calloc(1,sizeof(TYPE));
  *self = value;
  return self; %}
#endif
%{}

static void delete_##NAME(TYPE *self) { %}
#ifdef __cplusplus
%{  if (self) delete self; %}
#else
%{  if (self) free(self); %}
#endif
%{}

static void NAME ##_assign(TYPE *self, TYPE value) {
  *self = value;
}

static TYPE NAME ##_value(TYPE *self) {
  return *self;
}
%}

TYPE *new_##NAME();
TYPE *copy_##NAME(TYPE value);
void  delete_##NAME(TYPE *self);
void  NAME##_assign(TYPE *self, TYPE value);
TYPE  NAME##_value(TYPE *self);

%enddef

/* -----------------------------------------------------------------------------
 * %pointer_cast(type1,type2,name)
 *
 * Generates a pointer casting function.
 * ----------------------------------------------------------------------------- */

%define %pointer_cast(TYPE1,TYPE2,NAME)
%inline %{
TYPE2 NAME(TYPE1 x) {
   return (TYPE2) x;
}
%}
%enddef
