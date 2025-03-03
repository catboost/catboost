/* -----------------------------------------------------------------------------
 * error manipulation
 * ----------------------------------------------------------------------------- */

SWIGINTERN const char*
SWIG_Perl_ErrorType(int code) {
  switch(code) {
  case SWIG_MemoryError:
    return "MemoryError";
  case SWIG_IOError:
    return "IOError";
  case SWIG_RuntimeError:
    return "RuntimeError";
  case SWIG_IndexError:
    return "IndexError";
  case SWIG_TypeError:
    return "TypeError";
  case SWIG_DivisionByZero:
    return "ZeroDivisionError";
  case SWIG_OverflowError:
    return "OverflowError";
  case SWIG_SyntaxError:
    return "SyntaxError";
  case SWIG_ValueError:
    return "ValueError";
  case SWIG_SystemError:
    return "SystemError";
  case SWIG_AttributeError:
    return "AttributeError";
  default:
    return "RuntimeError";
  }
}

