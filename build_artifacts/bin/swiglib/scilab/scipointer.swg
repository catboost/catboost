/*
 * POINTER
 */
%fragment("SWIG_ConvertPtr", "header") {
#define SWIG_ConvertPtr(scilabValue, voidPointer, pointerDescriptor, flags) SwigScilabPtrToObject(pvApiCtx, scilabValue, voidPointer, pointerDescriptor, flags, SWIG_Scilab_GetFuncName())
}

%fragment("SWIG_NewPointerObj", "header") {
#define SWIG_NewPointerObj(pointer, pointerDescriptor, flags) SwigScilabPtrFromObject(pvApiCtx, SWIG_Scilab_GetOutputPosition(), pointer, pointerDescriptor, flags, NULL)
}

/*
 * FUNCTION POINTER
 */
%fragment("SWIG_ConvertFunctionPtr", "header") {
#define SWIG_ConvertFunctionPtr(scilabValue, voidPointer, pointerDescriptor) SwigScilabPtrToObject(pvApiCtx, scilabValue, voidPointer, pointerDescriptor, 0, SWIG_Scilab_GetFuncName())
}

%fragment("SWIG_NewFunctionPtrObj", "header") {
#define SWIG_NewFunctionPtrObj(pointer, pointerDescriptor) SwigScilabPtrFromObject(pvApiCtx, SWIG_Scilab_GetOutputPosition(), pointer, pointerDescriptor, 0, NULL)
}
// No fragment used here, the functions "SwigScilabPtrToObject" and "SwigScilabPtrFromObject" are defined in sciruntime.swg

/*
 * C++ member pointers, ie, member methods
 */
%fragment("SWIG_NewMemberObj", "header") {
#define SWIG_NewMemberObj(ptr, sz, tp) SWIG_Scilab_NewMemberObj(pvApiCtx, $result, ptr, sz, tp)
}
%fragment("SWIG_ConvertMember", "header") {
#define SWIG_ConvertMember(obj, ptr, sz, ty) SWIG_Scilab_ConvertPacked(pvApiCtx, obj, ptr, sz, ty, SWIG_Scilab_GetFuncName())
}
