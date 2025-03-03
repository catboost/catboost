/* -------------------------------------------------------------------------
 *  Special user directives
 * ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/*
  Implicit Conversion using the C++ constructor mechanism
*/

#define %implicitconv      %feature("implicitconv") 
#define %noimplicitconv    %feature("implicitconv", "0")
#define %clearimplicitconv %feature("implicitconv", "")


/* ------------------------------------------------------------------------- */
/* 
   %extend_smart_pointer extend the smart pointer support.

   For example, if you have a smart pointer as:
	    
     template <class Type> class RCPtr {
     public:
       ...
       RCPtr(Type *p);
   	Type * operator->() const;
   	...
     };
     
   you use the %extend_smart_pointer directive as:
   
     %extend_smart_pointer(RCPtr<A>);
     %template(RCPtr_A)  RCPtr<A>;
   
   then, if you have something like:

     RCPtr<A> make_ptr();
     int foo(A *);

   you can do the following:

     a = make_ptr();
     b = foo(a);

   ie, swig will accept a RCPtr<A> object where a 'A *' is
   expected.

   Also, when using vectors
   
     %extend_smart_pointer(RCPtr<A>);
     %template(RCPtr_A) RCPtr<A>;
     %template(vector_A) std::vector<RCPtr<A> >;
   	
   you can type

     a = A();
     v = vector_A(2)
     v[0] = a

   ie, an 'A *' object is accepted, via implicit conversion, 
   where a RCPtr<A> object is expected. Additionally

     x = v[0]

   returns (and sets 'x' as) a copy of v[0], making reference
   counting possible and consistent.
*/

%define %extend_smart_pointer(Type...)
%implicitconv Type;
%apply const SWIGTYPE& SMARTPOINTER { const Type& };
%apply SWIGTYPE SMARTPOINTER { Type };
%enddef
