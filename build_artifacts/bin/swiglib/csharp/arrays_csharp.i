/* -----------------------------------------------------------------------------
 * arrays_csharp.i
 *
 * This file contains a two approaches to marshaling arrays. The first uses
 * default p/invoke marshaling and the second uses pinning of the arrays.
 *
 * Default marshaling approach
 * ----------------------------
 * Array typemaps using default p/invoke marshaling. The data is copied to a separately
 * allocated buffer when passing over the managed-native boundary.
 * 
 * There are separate typemaps for in, out and inout arrays to enable avoiding 
 * unnecessary copying.
 * 
 * Example usage:
 *
 *   %include "arrays_csharp.i"
 *   %apply int INPUT[]  { int* sourceArray }
 *   %apply int OUTPUT[] { int* targetArray }
 *   void myArrayCopy( int* sourceArray, int* targetArray, int nitems );
 *
 *   %apply int INOUT[] { int* array1, int *array2 }
 *   void myArraySwap( int* array1, int* array2, int nitems );
 *
 * If handling large arrays you should consider using the pinning array typemaps
 * described next.
 *
 * Pinning approach
 * ----------------
 * Array typemaps using pinning. These typemaps pin the managed array given
 * as parameter and pass a pointer to it to the c/c++ side. This is very
 * efficient as no copying is done (unlike in the default array marshaling),
 * but it makes garbage collection more difficult. When considering using
 * these typemaps, think carefully whether you have callbacks that may cause
 * the control to re-enter the managed side from within the call (and produce
 * garbage for the gc) or whether other threads may produce enough garbage to 
 * trigger gc while the call is being executed. In those cases it may be
 * wiser to use the default marshaling typemaps.
 * 
 * Please note that when using fixed arrays, you have to mark your corresponding 
 * module class method unsafe using 
 * %csmethodmodifiers "public unsafe"
 * (the visibility of the method is up to you).
 *
 * Example usage:
 *
 *   %include "arrays_csharp.i"
 *   %apply int FIXED[] { int* sourceArray, int *targetArray }
 *   %csmethodmodifiers myArrayCopy "public unsafe";
 *   void myArrayCopy( int *sourceArray, int* targetArray, int nitems );
 *
 * ----------------------------------------------------------------------------- */

%define CSHARP_ARRAYS( CTYPE, CSTYPE )

// input only arrays

%typemap(ctype)   CTYPE INPUT[] "CTYPE*"
%typemap(cstype)  CTYPE INPUT[] "CSTYPE[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.In, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray)]") CTYPE INPUT[] "CSTYPE[]"
%typemap(csin)    CTYPE INPUT[] "$csinput"

%typemap(in)      CTYPE INPUT[] "$1 = $input;"
%typemap(freearg) CTYPE INPUT[] ""
%typemap(argout)  CTYPE INPUT[] ""

// output only arrays

%typemap(ctype)   CTYPE OUTPUT[] "CTYPE*"
%typemap(cstype)  CTYPE OUTPUT[] "CSTYPE[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.Out, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray)]") CTYPE OUTPUT[] "CSTYPE[]"
%typemap(csin)    CTYPE OUTPUT[] "$csinput"

%typemap(in)      CTYPE OUTPUT[] "$1 = $input;"
%typemap(freearg) CTYPE OUTPUT[] ""
%typemap(argout)  CTYPE OUTPUT[] ""

// inout arrays

%typemap(ctype)   CTYPE INOUT[] "CTYPE*"
%typemap(cstype)  CTYPE INOUT[] "CSTYPE[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.In, global::System.Runtime.InteropServices.Out, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray)]") CTYPE INOUT[] "CSTYPE[]"
%typemap(csin)    CTYPE INOUT[] "$csinput"

%typemap(in)      CTYPE INOUT[] "$1 = $input;"
%typemap(freearg) CTYPE INOUT[] ""
%typemap(argout)  CTYPE INOUT[] ""

%enddef // CSHARP_ARRAYS

CSHARP_ARRAYS(signed char, sbyte)
CSHARP_ARRAYS(unsigned char, byte)
CSHARP_ARRAYS(short, short)
CSHARP_ARRAYS(unsigned short, ushort)
CSHARP_ARRAYS(int, int)
CSHARP_ARRAYS(unsigned int, uint)
// FIXME - on Unix 64 bit, long is 8 bytes but is 4 bytes on Windows 64 bit.
//         How can this be handled sensibly?
//         See e.g. http://www.xml.com/ldd/chapter/book/ch10.html
CSHARP_ARRAYS(long, int)
CSHARP_ARRAYS(unsigned long, uint)
CSHARP_ARRAYS(long long, long)
CSHARP_ARRAYS(unsigned long long, ulong)
CSHARP_ARRAYS(float, float)
CSHARP_ARRAYS(double, double)

// By default C# will marshal bools as 4 bytes
// UnmanagedType.I1 will change this to 1 byte
// FIXME - When running on mono ArraySubType appears to be ignored and bools will be marshalled as 4-byte
// https://github.com/mono/mono/issues/15592

// input only arrays
%typemap(ctype)   bool INPUT[] "bool*"
%typemap(cstype)  bool INPUT[] "bool[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.In, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray,ArraySubType=System.Runtime.InteropServices.UnmanagedType.I1)]") bool INPUT[] "bool[]"
%typemap(csin)    bool INPUT[] "$csinput"

%typemap(in)      bool INPUT[] %{
$1 = $input;
%}
%typemap(freearg) bool INPUT[] ""
%typemap(argout)  bool INPUT[] ""

// output only arrays
%typemap(ctype)   bool OUTPUT[] "bool*"
%typemap(cstype)  bool OUTPUT[] "bool[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.Out, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray,ArraySubType=System.Runtime.InteropServices.UnmanagedType.I1)]") bool OUTPUT[] "bool[]"
%typemap(csin)    bool OUTPUT[] "$csinput"

%typemap(in)      bool OUTPUT[] %{
$1 = $input;
%}
%typemap(freearg) bool OUTPUT[] ""
%typemap(argout)  bool OUTPUT[] ""

// inout arrays
%typemap(ctype)   bool INOUT[] "bool*"
%typemap(cstype)  bool INOUT[] "bool[]"
%typemap(imtype, inattributes="[global::System.Runtime.InteropServices.In, global::System.Runtime.InteropServices.Out, global::System.Runtime.InteropServices.MarshalAs(global::System.Runtime.InteropServices.UnmanagedType.LPArray,ArraySubType=System.Runtime.InteropServices.UnmanagedType.I1)]") bool INOUT[] "bool[]"
%typemap(csin)    bool INOUT[] "$csinput"

%typemap(in)      bool INOUT[] %{
$1 = $input;
%}
%typemap(freearg) bool INOUT[] ""
%typemap(argout)  bool INOUT[] ""


%define CSHARP_ARRAYS_FIXED( CTYPE, CSTYPE )

%typemap(ctype)   CTYPE FIXED[] "CTYPE*"
%typemap(imtype)  CTYPE FIXED[] "global::System.IntPtr"
%typemap(cstype)  CTYPE FIXED[] "CSTYPE[]"
%typemap(csin,
           pre=       "    fixed ( CSTYPE* swig_ptrTo_$csinput = $csinput ) {",
           terminator="    }") 
                  CTYPE FIXED[] "(global::System.IntPtr)swig_ptrTo_$csinput"

%typemap(in)      CTYPE FIXED[] "$1 = $input;"
%typemap(freearg) CTYPE FIXED[] ""
%typemap(argout)  CTYPE FIXED[] ""


%enddef // CSHARP_ARRAYS_FIXED

CSHARP_ARRAYS_FIXED(signed char, sbyte)
CSHARP_ARRAYS_FIXED(unsigned char, byte)
CSHARP_ARRAYS_FIXED(short, short)
CSHARP_ARRAYS_FIXED(unsigned short, ushort)
CSHARP_ARRAYS_FIXED(int, int)
CSHARP_ARRAYS_FIXED(unsigned int, uint)
CSHARP_ARRAYS_FIXED(long, int)
CSHARP_ARRAYS_FIXED(unsigned long, uint)
CSHARP_ARRAYS_FIXED(long long, long)
CSHARP_ARRAYS_FIXED(unsigned long long, ulong)
CSHARP_ARRAYS_FIXED(float, float)
CSHARP_ARRAYS_FIXED(double, double)
CSHARP_ARRAYS_FIXED(bool, bool)

