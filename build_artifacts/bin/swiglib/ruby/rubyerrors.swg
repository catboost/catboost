/* -----------------------------------------------------------------------------
 * error manipulation
 * ----------------------------------------------------------------------------- */


/* Define some additional error types */
#define SWIG_ObjectPreviouslyDeletedError  -100


/* Define custom exceptions for errors that do not map to existing Ruby
   exceptions.  Note this only works for C++ since a global cannot be
   initialized by a function in C.  For C, fallback to rb_eRuntimeError.*/

SWIGINTERN VALUE 
getNullReferenceError(void) {
  static int init = 0;
  static VALUE rb_eNullReferenceError ;
  if (!init) {
    init = 1;
    rb_eNullReferenceError = rb_define_class("NullReferenceError", rb_eRuntimeError);
  }
  return rb_eNullReferenceError;
} 

SWIGINTERN VALUE 
getObjectPreviouslyDeletedError(void) {
  static int init = 0;
  static VALUE rb_eObjectPreviouslyDeleted ;
  if (!init) {
    init = 1;
    rb_eObjectPreviouslyDeleted = rb_define_class("ObjectPreviouslyDeleted", rb_eRuntimeError);
  }
  return rb_eObjectPreviouslyDeleted;
} 


SWIGINTERN VALUE
SWIG_Ruby_ErrorType(int SWIG_code) {
  VALUE type;
  switch (SWIG_code) {
  case SWIG_MemoryError:
    type = rb_eNoMemError;
    break;
  case SWIG_IOError:
    type = rb_eIOError;
    break;
  case SWIG_RuntimeError:
    type = rb_eRuntimeError;
    break;
  case SWIG_IndexError:
    type = rb_eIndexError;
    break;
  case SWIG_TypeError:
    type = rb_eTypeError;
    break;
  case SWIG_DivisionByZero:
    type = rb_eZeroDivError;
    break;
  case SWIG_OverflowError:
    type = rb_eRangeError;
    break;
  case SWIG_SyntaxError:
    type = rb_eSyntaxError;
    break;
  case SWIG_ValueError:
    type = rb_eArgError;
    break;
  case SWIG_SystemError:
    type = rb_eFatal;
    break;
  case SWIG_AttributeError:
    type = rb_eRuntimeError;
    break;
  case SWIG_NullReferenceError:
    type = getNullReferenceError();
    break;
  case SWIG_ObjectPreviouslyDeletedError:
    type = getObjectPreviouslyDeletedError();
    break;
  case SWIG_UnknownError:
    type = rb_eRuntimeError;
    break;
  default:
    type = rb_eRuntimeError;
  }
  return type;
}


/* This function is called when a user inputs a wrong argument to
   a method.
 */
SWIGINTERN 
const char* Ruby_Format_TypeError( const char* msg,
				   const char* type, 
				   const char* name, 
				   const int argn,
				   VALUE input )
{
  char buf[128];
  VALUE str;
  VALUE asStr;
  if ( msg && *msg )
    {
      str = rb_str_new2(msg);
    }
  else
    {
      str = rb_str_new(NULL, 0);
    }

  str = rb_str_cat2( str, "Expected argument " );
  sprintf( buf, "%d of type ", argn-1 );
  str = rb_str_cat2( str, buf );
  str = rb_str_cat2( str, type );
  str = rb_str_cat2( str, ", but got " );
  str = rb_str_cat2( str, rb_obj_classname(input) );
  str = rb_str_cat2( str, " " );
  asStr = rb_inspect(input);
  if ( RSTRING_LEN(asStr) > 30 )
    {
      str = rb_str_cat( str, StringValuePtr(asStr), 30 );
      str = rb_str_cat2( str, "..." );
    }
  else
    {
      str = rb_str_append( str, asStr );
    }

  if ( name )
    {
      str = rb_str_cat2( str, "\n\tin SWIG method '" );
      str = rb_str_cat2( str, name );
      str = rb_str_cat2( str, "'" );
    }

  return StringValuePtr( str );
}

/* This function is called when an overloaded method fails */
SWIGINTERN 
void Ruby_Format_OverloadedError(
				 const int argc,
				 const int maxargs,
				 const char* method, 
				 const char* prototypes 
				 )
{
  const char* msg = "Wrong # of arguments";
  if ( argc <= maxargs ) msg = "Wrong arguments";
  rb_raise(rb_eArgError,"%s for overloaded method '%s'.\n"  
	   "Possible C/C++ prototypes are:\n%s",
	   msg, method, prototypes);
}
