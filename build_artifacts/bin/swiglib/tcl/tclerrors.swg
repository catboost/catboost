/* -----------------------------------------------------------------------------
 * error manipulation
 * ----------------------------------------------------------------------------- */

SWIGINTERN const char*
SWIG_Tcl_ErrorType(int code) {
  const char* type = 0;
  switch(code) {
  case SWIG_MemoryError:
    type = "MemoryError";
    break;
  case SWIG_IOError:
    type = "IOError";
    break;
  case SWIG_RuntimeError:
    type = "RuntimeError";
    break;
  case SWIG_IndexError:
    type = "IndexError";
    break;
  case SWIG_TypeError:
    type = "TypeError";
    break;
  case SWIG_DivisionByZero:
    type = "ZeroDivisionError";
    break;
  case SWIG_OverflowError:
    type = "OverflowError";
    break;
  case SWIG_SyntaxError:
    type = "SyntaxError";
    break;
  case SWIG_ValueError:
    type = "ValueError";
    break;
  case SWIG_SystemError:
    type = "SystemError";
    break;
  case SWIG_AttributeError:
    type = "AttributeError";
    break;
  default:
    type = "RuntimeError";
  }
  return type;
}


SWIGINTERN void
SWIG_Tcl_SetErrorObj(Tcl_Interp *interp, const char *ctype, Tcl_Obj *obj)
{
  Tcl_ResetResult(interp);
  Tcl_SetObjResult(interp, obj);
  Tcl_SetErrorCode(interp, "SWIG", ctype, NULL);
}

SWIGINTERN void
SWIG_Tcl_SetErrorMsg(Tcl_Interp *interp, const char *ctype, const char *mesg)
{
  Tcl_ResetResult(interp);
  Tcl_SetErrorCode(interp, "SWIG", ctype, NULL);
  Tcl_AppendResult(interp, ctype, " ", mesg, NULL);
  /*
  Tcl_AddErrorInfo(interp, ctype);
  Tcl_AddErrorInfo(interp, " ");
  Tcl_AddErrorInfo(interp, mesg);
  */
}

SWIGINTERNINLINE void
SWIG_Tcl_AddErrorMsg(Tcl_Interp *interp, const char* mesg)
{
  Tcl_AddErrorInfo(interp, mesg);
}


