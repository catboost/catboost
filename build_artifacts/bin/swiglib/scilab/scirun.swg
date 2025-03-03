/* -----------------------------------------------------------------------------
 * Scilab support runtime
 * -----------------------------------------------------------------------------*/

/* Scilab version macro */

#include "version.h"
#define SWIG_SCILAB_VERSION (SCI_VERSION_MAJOR * 100) + (SCI_VERSION_MINOR * 10) + SCI_VERSION_MAINTENANCE

/* Scilab standard headers */

#ifdef __cplusplus
extern "C" {
#endif
#include "api_scilab.h"
#if SWIG_SCILAB_VERSION < 540
#define __USE_DEPRECATED_STACK_FUNCTIONS__
#include "stack-c.h"
#endif
#if SWIG_SCILAB_VERSION < 600
#include "MALLOC.h"
#endif
#include "Scierror.h"
#include "localization.h"
#include "freeArrayOfString.h"
#include <sci_gateway.h>
#include <mex.h>
#ifdef __cplusplus
}
#endif

/* Gateway signature */

#if SWIG_SCILAB_VERSION >= 600
#define SWIG_GatewayParameters char* fname, void *pvApiCtx
#define SWIG_GatewayArguments fname, pvApiCtx
#else
#define SWIG_GatewayParameters char* fname, unsigned long fname_len
#define SWIG_GatewayArguments fname, fname_len
#endif

/* Function name management functions */

#include <stdlib.h>
static char *SwigFuncName = NULL;
static char *SWIG_Scilab_GetFuncName(void) {
  return SwigFuncName;
}
static void SWIG_Scilab_SetFuncName(char *funcName) {
  if (SwigFuncName != NULL) {
    free(SwigFuncName);
    SwigFuncName = NULL;
  }
  if (funcName) {
    SwigFuncName = (char *)malloc(strlen(funcName) + 1);
    if (SwigFuncName)
      strcpy(SwigFuncName, funcName);
  }
}

/* Api context management functions */

#if SWIG_SCILAB_VERSION >= 600
static void *pvApiCtx = NULL;
static void SWIG_Scilab_SetApiContext(void *apiCtx) {
  pvApiCtx = apiCtx;
}
#else
#define SWIG_Scilab_SetApiContext(apiCtx)
#endif

/* Argument management functions */

#if SWIG_SCILAB_VERSION >= 540
#define SWIG_CheckInputArgument(pvApiCtx, minInputArgument, maxInputArgument) CheckInputArgument(pvApiCtx, minInputArgument, maxInputArgument)
#define SWIG_CheckInputArgumentAtLeast(pvApiCtx, minInputArgument) CheckInputArgumentAtLeast(pvApiCtx, minInputArgument)
#define SWIG_CheckOutputArgument(pvApiCtx, minOutputArgument, maxOutputArgument) CheckOutputArgument(pvApiCtx, minOutputArgument, maxOutputArgument)
#define SWIG_NbInputArgument(pvApiCtx) nbInputArgument(pvApiCtx)
#define SWIG_AssignOutputArgument(pvApiCtx, outputArgumentPos, argumentPos) AssignOutputVariable(pvApiCtx, outputArgumentPos) = argumentPos
#else
#define SWIG_CheckInputArgument(pvApiCtx, minInputArgument, maxInputArgument) CheckRhs(minInputArgument, maxInputArgument)
#define SWIG_CheckInputArgumentAtLeast(pvApiCtx, minInputArgument) CheckRhs(minInputArgument, 256)
#define SWIG_CheckOutputArgument(pvApiCtx, minOutputArgument, maxOutputArgument) CheckLhs(minOutputArgument, maxOutputArgument)
#define SWIG_NbInputArgument(pvApiCtx) Rhs
#define SWIG_AssignOutputArgument(pvApiCtx, outputArgumentPos, argumentPos) LhsVar(outputArgumentPos) = argumentPos
#endif

typedef int SwigSciObject;

static int SwigOutputPosition = -1;
static int SWIG_Scilab_GetOutputPosition(void) {
  return SwigOutputPosition;
}
static void SWIG_Scilab_SetOutputPosition(int outputPosition) {
  SwigOutputPosition = outputPosition;
}

SWIGRUNTIME int
SWIG_Scilab_SetOutput(void *pvApiCtx, SwigSciObject output) {
  int outputPosition = SWIG_Scilab_GetOutputPosition();
  if (outputPosition < 0)
      return SWIG_ERROR;
  SWIG_AssignOutputArgument(pvApiCtx, outputPosition,
    SWIG_NbInputArgument(pvApiCtx) + outputPosition);
  return SWIG_OK;
}

/* Error functions */

#define SCILAB_API_ARGUMENT_ERROR 999

SWIGINTERN const char*
SWIG_Scilab_ErrorType(int code) {
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
#define SWIG_ErrorType(code) SWIG_Scilab_ErrorType(code)

#ifndef SWIG_SCILAB_ERROR
#define SWIG_SCILAB_ERROR 20000
#endif

SWIGINTERN void
SWIG_Scilab_Error(int code, const char *msg) {
  Scierror(SWIG_SCILAB_ERROR - code, _("SWIG/Scilab: %s: %s\n"), SWIG_Scilab_ErrorType(code), msg);
}

#define SWIG_Error(code, msg) SWIG_Scilab_Error(code, msg)

#define SWIG_fail return SWIG_ERROR;

SWIGRUNTIME void
SWIG_Scilab_Raise_Ex(const char *obj, const char *type, swig_type_info *descriptor) {
  if (type) {
    if (obj)
      Scierror(SWIG_SCILAB_ERROR, "SWIG/Scilab: Exception (%s) occurred: %s\n", type, obj);
    else
      Scierror(SWIG_SCILAB_ERROR, "SWIG/Scilab: Exception (%s) occurred.\n", type);
  }
}

SWIGRUNTIME void
SWIG_Scilab_Raise(const int obj, const char *type, swig_type_info *descriptor) {
  Scierror(SWIG_SCILAB_ERROR, "SWIG/Scilab: Exception (%s) occurred.\n", type);
}

/* Module initialization */

static int swig_module_initialized = 0;

SWIGRUNTIME int
SWIG_Module_Initialized() {
  return swig_module_initialized;
}

/* Pointer conversion functions */

SWIGRUNTIME swig_type_info *
SWIG_Scilab_TypeQuery(const char *name);

SWIGINTERN int
SwigScilabCheckPtr(void *pvApiCtx, int iVar, swig_type_info *descriptor, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  int iType = 0;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piAddrVar, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (iType == sci_mlist) {
    int iItemCount = 0;
    void *pvTypeinfo = NULL;

    sciErr = getListItemNumber(pvApiCtx, piAddrVar, &iItemCount);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iItemCount < 3) {
      return SWIG_ERROR;
    }

    sciErr = getPointerInList(pvApiCtx, piAddrVar, 2, &pvTypeinfo);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if (descriptor) {
      swig_cast_info *cast = SWIG_TypeCheck(SWIG_TypeName((swig_type_info*)pvTypeinfo), descriptor);
      return (cast != NULL);
    }
    else {
      return SWIG_ERROR;
    }
  }
  else {
    return (iType == sci_pointer);
  }
}

SWIGINTERN int
SwigScilabPtrToObject(void *pvApiCtx, int iVar, void **pvObj, swig_type_info *descriptor, int flags, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  int iType = 0;
  void *pvPtr = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piAddrVar, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (iType == sci_mlist) {
    int iItemCount = 0;
    void *pvTypeinfo = NULL;

    sciErr = getListItemNumber(pvApiCtx, piAddrVar, &iItemCount);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iItemCount < 3) {
      return SWIG_ERROR;
    }

    sciErr = getPointerInList(pvApiCtx, piAddrVar, 2, &pvTypeinfo);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    sciErr = getPointerInList(pvApiCtx, piAddrVar, 3, &pvPtr);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if (pvPtr) {
      if (descriptor) {
        swig_cast_info *cast = SWIG_TypeCheck(SWIG_TypeName((swig_type_info *)pvTypeinfo), descriptor);
        if (cast) {
          int newmemory = 0;
          pvPtr = SWIG_TypeCast(cast, pvPtr, &newmemory);
          // TODO newmemory
        }
        else {
          return SWIG_ERROR;
        }
      }
    }
  }
  else if (iType == sci_pointer) {
    sciErr = getPointer(pvApiCtx, piAddrVar, &pvPtr);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    return SWIG_ERROR;
  }

  if (pvObj) {
    *pvObj = pvPtr;
    if (pvPtr)
      return SWIG_OK;
    else
      return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }
  else {
    return SWIG_ERROR;
  }
}

SWIGRUNTIMEINLINE int
SwigScilabPtrFromObject(void *pvApiCtx, int iVarOut, void *pvObj, swig_type_info *descriptor, int flags, const char *pstTypeName) {
  SciErr sciErr;

  if (descriptor) {
    int *piMListAddr = NULL;

    sciErr = createMList(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, 3, &piMListAddr);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if (pstTypeName == NULL) {
      pstTypeName = SWIG_TypeName(descriptor);
    }

    sciErr = createMatrixOfStringInList(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, piMListAddr, 1, 1, 1, &pstTypeName);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    sciErr = createPointerInList(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, piMListAddr, 2, descriptor);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    sciErr = createPointerInList(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, piMListAddr, 3, pvObj);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    sciErr = createPointer(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, pvObj);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }

  return SWIG_OK;
}

/* Pointer argument conversions */


SWIGRUNTIME int
SWIG_Scilab_ConvertPacked(void *pvApiCtx, int iVar, void *ptr, int sz, swig_type_info *ty, char *fname) {
  swig_cast_info *tc;
  int *piAddrVar = NULL;
  char *pstString = NULL;
  char *pstStringPtr = NULL;
  SciErr sciErr;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (getAllocatedSingleString(pvApiCtx, piAddrVar, &pstString)) {
    return SWIG_ERROR;
  }

  /* Pointer values must start with leading underscore */
  if (*pstString != '_') {
    freeAllocatedSingleString(pstString);
    return SWIG_ERROR;
  }

  pstStringPtr = pstString;
  pstStringPtr++;
  pstStringPtr = (char*)SWIG_UnpackData(pstStringPtr, ptr, sz);

  if (ty) {
    if (!pstStringPtr) {
      freeAllocatedSingleString(pstString);
      return SWIG_ERROR;
    }
    tc = SWIG_TypeCheck(pstStringPtr, ty);
    if (!tc) {
      freeAllocatedSingleString(pstString);
      return SWIG_ERROR;
    }
  }

  freeAllocatedSingleString(pstString);
  return SWIG_OK;
}

SWIGRUNTIME int
SWIG_Scilab_NewMemberObj(void *pvApiCtx, int iVarOut, void *ptr, int sz, swig_type_info *type) {
  char result[1024];
  char *r = result;

  if ((2*sz + 1 + strlen(type->name)) > 1000) {
    return SWIG_ERROR;
  }
  *(r++) = '_';
  r = SWIG_PackData(r, ptr, sz);
  strcpy(r, type->name);

  if (createSingleString(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, &result[0]))
    return SWIG_ERROR;

  return SWIG_OK;
}




/*
 *  Pointer utility functions
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C"
#endif
int SWIG_this(SWIG_GatewayParameters) {
  void *ptrValue = NULL;
  if (SwigScilabPtrToObject(pvApiCtx, 1, &ptrValue, NULL, 0, fname) == SWIG_OK) {
    SWIG_Scilab_SetOutputPosition(1);
    return SWIG_Scilab_SetOutput(pvApiCtx,
      createScalarDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + 1,
        (double)(uintptr_t)ptrValue));
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The value is not a pointer.\n"), fname, 1);
    return SWIG_ERROR;
  }
}

#ifdef __cplusplus
extern "C"
#endif
int SWIG_ptr(SWIG_GatewayParameters) {
  if (SWIG_NbInputArgument(pvApiCtx) > 0) {
    SciErr sciErr;
    int *piAddrVar1 = NULL;
    int iTypeVar1 = 0;
    char *pstInputPtrTypeName = NULL;
    char *pstOutputMListTypeName = NULL;
    if (SWIG_NbInputArgument(pvApiCtx) > 2) {
      int *piAddrVar2 = NULL;
      int *piAddrVar3 = NULL;
      sciErr = getVarAddressFromPosition(pvApiCtx, 2, &piAddrVar2);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (getAllocatedSingleString(pvApiCtx, piAddrVar2, &pstInputPtrTypeName)) {
        return SWIG_ERROR;
      }
      sciErr = getVarAddressFromPosition(pvApiCtx, 3, &piAddrVar3);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (getAllocatedSingleString(pvApiCtx, piAddrVar3, &pstOutputMListTypeName)) {
        return SWIG_ERROR;
      }
    }

    sciErr = getVarAddressFromPosition(pvApiCtx, 1, &piAddrVar1);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    sciErr = getVarType(pvApiCtx, piAddrVar1, &iTypeVar1);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if ((iTypeVar1 == sci_pointer) || (iTypeVar1 == sci_mlist)) {
      void *ptrValue = NULL;
      if (SwigScilabPtrToObject(pvApiCtx, 1, &ptrValue, SWIG_Scilab_TypeQuery(pstInputPtrTypeName), 0, (char *) "SWIG_ptr") == SWIG_OK) {
        SWIG_Scilab_SetOutputPosition(1);
        return SWIG_Scilab_SetOutput(pvApiCtx,
          SwigScilabPtrFromObject(pvApiCtx, 1, ptrValue, SWIG_Scilab_TypeQuery(pstInputPtrTypeName), 0, pstOutputMListTypeName));
      }
      else {
        return SWIG_ERROR;
      }
    }
    else if (iTypeVar1 == sci_matrix) {
      double dValue = 0;
      if (getScalarDouble(pvApiCtx, piAddrVar1, &dValue) == 0) {
        if (dValue != (uintptr_t)dValue) {
          Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The double value cannot be converted to a pointer.\n"), fname, 1);
          return SWIG_ValueError;
        }
        if ((dValue < 0) || (dValue > ULONG_MAX)) {
          Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Overflow error for input argument #%d: The double value cannot be converted to a pointer.\n"), fname, 1);
          return SWIG_OverflowError;
        }
        SWIG_Scilab_SetOutputPosition(1);
        return SWIG_Scilab_SetOutput(pvApiCtx,
          SwigScilabPtrFromObject(pvApiCtx, 1, (void *) (uintptr_t)dValue, SWIG_Scilab_TypeQuery(pstInputPtrTypeName), 0, pstOutputMListTypeName));
      }
      else {
        return SWIG_TypeError;
      }
    }
    else {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A mlist, pointer or a double expected.\n"), (char *) "SWIG_ptr", 1);
      return SWIG_TypeError;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: A mlist, pointer, or a double expected.\n"), "SWIG_ptr", 1);
    return SWIG_TypeError;
  }
}
