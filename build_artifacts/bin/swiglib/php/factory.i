/*
  Implement a more natural wrap for factory methods, for example, if
  you have:

  ----  geometry.h --------
       struct Geometry {                          
         enum GeomType{			     
           POINT,				     
           CIRCLE				     
         };					     
         					     
         virtual ~Geometry() {}    		     
         virtual int draw() = 0;
	 
	 //
	 // Factory method for all the Geometry objects
	 //
         static Geometry *create(GeomType i);     
       };					     
       					     
       struct Point : Geometry  {		     
         int draw() { return 1; }		     
         double width() { return 1.0; }    	     
       };					     
       					     
       struct Circle : Geometry  {		     
         int draw() { return 2; }		     
         double radius() { return 1.5; }          
       }; 					     
       
       //
       // Factory method for all the Geometry objects
       //
       Geometry *Geometry::create(GeomType type) {
         switch (type) {			     
         case POINT: return new Point();	     
         case CIRCLE: return new Circle(); 	     
         default: return 0;			     
         }					     
       }					    
  ----  geometry.h --------


  You can use the %factory with the Geometry::create method as follows:

    %newobject Geometry::create;
    %factory(Geometry *Geometry::create, Point, Circle);
    %include "geometry.h"

  and Geometry::create will return a 'Point' or 'Circle' instance
  instead of the plain 'Geometry' type. For example, in python:

    circle = Geometry.create(Geometry.CIRCLE)
    r = circle.radius()

  where circle is a Circle proxy instance.

  NOTES: remember to fully qualify all the type names and don't
  use %factory inside a namespace declaration, ie, instead of
  
     namespace Foo {
       %factory(Geometry *Geometry::create, Point, Circle);
     }

  use

     %factory(Foo::Geometry *Foo::Geometry::create, Foo::Point,  Foo::Circle);   

     
*/

/* for loop for macro with one argument */
%define %_formacro_1(macro, arg1,...)macro(arg1)
#if #__VA_ARGS__ != "__fordone__"
%_formacro_1(macro, __VA_ARGS__)
#endif
%enddef

/* for loop for macro with one argument */
%define %formacro_1(macro,...)%_formacro_1(macro,__VA_ARGS__,__fordone__)%enddef
%define %formacro(macro,...)%_formacro_1(macro,__VA_ARGS__,__fordone__)%enddef

/* for loop for macro with two arguments */
%define %_formacro_2(macro, arg1, arg2, ...)macro(arg1, arg2)
#if #__VA_ARGS__ != "__fordone__"
%_formacro_2(macro, __VA_ARGS__)
#endif
%enddef

/* for loop for macro with two arguments */
%define %formacro_2(macro,...)%_formacro_2(macro, __VA_ARGS__, __fordone__)%enddef

%define %_factory_dispatch(Type) 
if (!dcast) {
  Type *dobj = dynamic_cast<Type *>($1);
  if (dobj) {
    dcast = 1;
    SWIG_SetPointerZval(return_value, SWIG_as_voidptr(dobj),$descriptor(Type *), $owner);
  }   
}%enddef

%define %factory(Method,Types...)
%typemap(out) Method {
  int dcast = 0;
  %formacro(%_factory_dispatch, Types)
  if (!dcast) {
    SWIG_SetPointerZval(return_value, SWIG_as_voidptr($1),$descriptor, $owner);
  }
}%enddef
