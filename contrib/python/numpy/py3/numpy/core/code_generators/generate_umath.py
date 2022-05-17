import os
import re
import struct
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

Zero = "PyLong_FromLong(0)"
One = "PyLong_FromLong(1)"
True_ = "(Py_INCREF(Py_True), Py_True)"
False_ = "(Py_INCREF(Py_False), Py_False)"
None_ = object()
AllOnes = "PyLong_FromLong(-1)"
MinusInfinity = 'PyFloat_FromDouble(-NPY_INFINITY)'
ReorderableNone = "(Py_INCREF(Py_None), Py_None)"

# Sentinel value to specify using the full type description in the
# function name
class FullTypeDescr:
    pass

class FuncNameSuffix:
    """Stores the suffix to append when generating functions names.
    """
    def __init__(self, suffix):
        self.suffix = suffix

class TypeDescription:
    """Type signature for a ufunc.

    Attributes
    ----------
    type : str
        Character representing the nominal type.
    func_data : str or None or FullTypeDescr or FuncNameSuffix, optional
        The string representing the expression to insert into the data
        array, if any.
    in_ : str or None, optional
        The typecode(s) of the inputs.
    out : str or None, optional
        The typecode(s) of the outputs.
    astype : dict or None, optional
        If astype['x'] is 'y', uses PyUFunc_x_x_As_y_y/PyUFunc_xx_x_As_yy_y
        instead of PyUFunc_x_x/PyUFunc_xx_x.
    cfunc_alias : str or none, optional
        Appended to inner loop C function name, e.g., FLOAT_{cfunc_alias}. See make_arrays.
        NOTE: it doesn't support 'astype'
    simd: list
        Available SIMD ufunc loops, dispatched at runtime in specified order
        Currently only supported for simples types (see make_arrays)
    dispatch: str or None, optional
        Dispatch-able source name without its extension '.dispatch.c' that
        contains the definition of ufunc, dispatched at runtime depending on the
        specified targets of the dispatch-able source.
        NOTE: it doesn't support 'astype'
    """
    def __init__(self, type, f=None, in_=None, out=None, astype=None, cfunc_alias=None,
                 simd=None, dispatch=None):
        self.type = type
        self.func_data = f
        if astype is None:
            astype = {}
        self.astype_dict = astype
        if in_ is not None:
            in_ = in_.replace('P', type)
        self.in_ = in_
        if out is not None:
            out = out.replace('P', type)
        self.out = out
        self.cfunc_alias = cfunc_alias
        self.simd = simd
        self.dispatch = dispatch

    def finish_signature(self, nin, nout):
        if self.in_ is None:
            self.in_ = self.type * nin
        assert len(self.in_) == nin
        if self.out is None:
            self.out = self.type * nout
        assert len(self.out) == nout
        self.astype = self.astype_dict.get(self.type, None)

_fdata_map = dict(
    e='npy_%sf',
    f='npy_%sf',
    d='npy_%s',
    g='npy_%sl',
    F='nc_%sf',
    D='nc_%s',
    G='nc_%sl'
)

def build_func_data(types, f):
    func_data = [_fdata_map.get(t, '%s') % (f,) for t in types]
    return func_data

def TD(types, f=None, astype=None, in_=None, out=None, cfunc_alias=None,
       simd=None, dispatch=None):
    if f is not None:
        if isinstance(f, str):
            func_data = build_func_data(types, f)
        elif len(f) != len(types):
            raise ValueError("Number of types and f do not match")
        else:
            func_data = f
    else:
        func_data = (None,) * len(types)
    if isinstance(in_, str):
        in_ = (in_,) * len(types)
    elif in_ is None:
        in_ = (None,) * len(types)
    elif len(in_) != len(types):
        raise ValueError("Number of types and inputs do not match")
    if isinstance(out, str):
        out = (out,) * len(types)
    elif out is None:
        out = (None,) * len(types)
    elif len(out) != len(types):
        raise ValueError("Number of types and outputs do not match")
    tds = []
    for t, fd, i, o in zip(types, func_data, in_, out):
        # [(simd-name, list of types)]
        if simd is not None:
            simdt = [k for k, v in simd if t in v]
        else:
            simdt = []

        # [(dispatch file name without extension '.dispatch.c*', list of types)]
        if dispatch:
            dispt = ([k for k, v in dispatch if t in v]+[None])[0]
        else:
            dispt = None
        tds.append(TypeDescription(
            t, f=fd, in_=i, out=o, astype=astype, cfunc_alias=cfunc_alias,
            simd=simdt, dispatch=dispt
        ))
    return tds

class Ufunc:
    """Description of a ufunc.

    Attributes
    ----------
    nin : number of input arguments
    nout : number of output arguments
    identity : identity element for a two-argument function
    docstring : docstring for the ufunc
    type_descriptions : list of TypeDescription objects
    """
    def __init__(self, nin, nout, identity, docstring, typereso,
                 *type_descriptions, signature=None):
        self.nin = nin
        self.nout = nout
        if identity is None:
            identity = None_
        self.identity = identity
        self.docstring = docstring
        self.typereso = typereso
        self.type_descriptions = []
        self.signature = signature
        for td in type_descriptions:
            self.type_descriptions.extend(td)
        for td in self.type_descriptions:
            td.finish_signature(self.nin, self.nout)

# String-handling utilities to avoid locale-dependence.

import string
UPPER_TABLE = bytes.maketrans(bytes(string.ascii_lowercase, "ascii"),
                              bytes(string.ascii_uppercase, "ascii"))

def english_upper(s):
    """ Apply English case rules to convert ASCII strings to all upper case.

    This is an internal utility function to replace calls to str.upper() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "i".upper() != "I" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    uppered : str

    Examples
    --------
    >>> from numpy.lib.utils import english_upper
    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_upper(s)
    'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    >>> english_upper('')
    ''
    """
    uppered = s.translate(UPPER_TABLE)
    return uppered


#each entry in defdict is a Ufunc object.

#name: [string of chars for which it is defined,
#       string of characters using func interface,
#       tuple of strings giving funcs for data,
#       (in, out), or (instr, outstr) giving the signature as character codes,
#       identity,
#       docstring,
#       output specification (optional)
#       ]

chartoname = {
    '?': 'bool',
    'b': 'byte',
    'B': 'ubyte',
    'h': 'short',
    'H': 'ushort',
    'i': 'int',
    'I': 'uint',
    'l': 'long',
    'L': 'ulong',
    'q': 'longlong',
    'Q': 'ulonglong',
    'e': 'half',
    'f': 'float',
    'd': 'double',
    'g': 'longdouble',
    'F': 'cfloat',
    'D': 'cdouble',
    'G': 'clongdouble',
    'M': 'datetime',
    'm': 'timedelta',
    'O': 'OBJECT',
    # '.' is like 'O', but calls a method of the object instead
    # of a function
    'P': 'OBJECT',
}

noobj = '?bBhHiIlLqQefdgFDGmM'
all = '?bBhHiIlLqQefdgFDGOmM'

O = 'O'
P = 'P'
ints = 'bBhHiIlLqQ'
sints = 'bhilq'
uints = 'BHILQ'
times = 'Mm'
timedeltaonly = 'm'
intsO = ints + O
bints = '?' + ints
bintsO = bints + O
flts = 'efdg'
fltsO = flts + O
fltsP = flts + P
cmplx = 'FDG'
cmplxvec = 'FD'
cmplxO = cmplx + O
cmplxP = cmplx + P
inexact = flts + cmplx
inexactvec = 'fd'
noint = inexact+O
nointP = inexact+P
allP = bints+times+flts+cmplxP
nobool_or_obj = noobj[1:]
nobool_or_datetime = noobj[1:-1] + O # includes m - timedelta64
intflt = ints+flts
intfltcmplx = ints+flts+cmplx
nocmplx = bints+times+flts
nocmplxO = nocmplx+O
nocmplxP = nocmplx+P
notimes_or_obj = bints + inexact
nodatetime_or_obj = bints + inexact

# Find which code corresponds to int64.
int64 = ''
uint64 = ''
for code in 'bhilq':
    if struct.calcsize(code) == 8:
        int64 = code
        uint64 = english_upper(code)
        break

# This dictionary describes all the ufunc implementations, generating
# all the function names and their corresponding ufunc signatures.  TD is
# an object which expands a list of character codes into an array of
# TypeDescriptions.
defdict = {
'add':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.add'),
          'PyUFunc_AdditionTypeResolver',
          TD(notimes_or_obj, simd=[('avx2', ints)], dispatch=[('loops_arithm_fp', 'fdFD')]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'mM', 'M'),
          ],
          TD(O, f='PyNumber_Add'),
          ),
'subtract':
    Ufunc(2, 1, None, # Zero is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.subtract'),
          'PyUFunc_SubtractionTypeResolver',
          TD(ints + inexact, simd=[('avx2', ints)], dispatch=[('loops_arithm_fp', 'fdFD')]),
          [TypeDescription('M', FullTypeDescr, 'Mm', 'M'),
           TypeDescription('m', FullTypeDescr, 'mm', 'm'),
           TypeDescription('M', FullTypeDescr, 'MM', 'm'),
          ],
          TD(O, f='PyNumber_Subtract'),
          ),
'multiply':
    Ufunc(2, 1, One,
          docstrings.get('numpy.core.umath.multiply'),
          'PyUFunc_MultiplicationTypeResolver',
          TD(notimes_or_obj, simd=[('avx2', ints)], dispatch=[('loops_arithm_fp', 'fdFD')]),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'qm', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'dm', 'm'),
          ],
          TD(O, f='PyNumber_Multiply'),
          ),
#'divide' : aliased to true_divide in umathmodule.c:initumath
'floor_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.floor_divide'),
          'PyUFunc_DivisionTypeResolver',
          TD(ints, cfunc_alias='divide',
              dispatch=[('loops_arithmetic', 'bBhHiIlLqQ')]),
          TD(flts),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm'),
           TypeDescription('m', FullTypeDescr, 'md', 'm'),
           TypeDescription('m', FullTypeDescr, 'mm', 'q'),
          ],
          TD(O, f='PyNumber_FloorDivide'),
          ),
'true_divide':
    Ufunc(2, 1, None, # One is only a unit to the right, not the left
          docstrings.get('numpy.core.umath.true_divide'),
          'PyUFunc_TrueDivisionTypeResolver',
          TD(flts+cmplx, cfunc_alias='divide', dispatch=[('loops_arithm_fp', 'fd')]),
          [TypeDescription('m', FullTypeDescr, 'mq', 'm', cfunc_alias='divide'),
           TypeDescription('m', FullTypeDescr, 'md', 'm', cfunc_alias='divide'),
           TypeDescription('m', FullTypeDescr, 'mm', 'd', cfunc_alias='divide'),
          ],
          TD(O, f='PyNumber_TrueDivide'),
          ),
'conjugate':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.conjugate'),
          None,
          TD(ints+flts+cmplx, simd=[('avx2', ints), ('avx512f', cmplxvec)]),
          TD(P, f='conjugate'),
          ),
'fmod':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.fmod'),
          None,
          TD(ints),
          TD(flts, f='fmod', astype={'e': 'f'}),
          TD(P, f='fmod'),
          ),
'square':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.square'),
          None,
          TD(ints+inexact, simd=[('avx2', ints), ('avx512f', 'FD')], dispatch=[('loops_unary_fp', 'fd')]),
          TD(O, f='Py_square'),
          ),
'reciprocal':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.reciprocal'),
          None,
          TD(ints+inexact, simd=[('avx2', ints)], dispatch=[('loops_unary_fp', 'fd')]),
          TD(O, f='Py_reciprocal'),
          ),
# This is no longer used as numpy.ones_like, however it is
# still used by some internal calls.
'_ones_like':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath._ones_like'),
          'PyUFunc_OnesLikeTypeResolver',
          TD(noobj),
          TD(O, f='Py_get_one'),
          ),
'power':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.power'),
          None,
          TD(ints),
          TD(inexact, f='pow', astype={'e': 'f'}),
          TD(O, f='npy_ObjectPower'),
          ),
'float_power':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.float_power'),
          None,
          TD('dgDG', f='pow'),
          ),
'absolute':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.absolute'),
          'PyUFunc_AbsoluteTypeResolver',
          TD(bints+flts+timedeltaonly, dispatch=[('loops_unary_fp', 'fd')]),
          TD(cmplx, simd=[('avx512f', cmplxvec)], out=('f', 'd', 'g')),
          TD(O, f='PyNumber_Absolute'),
          ),
'_arg':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath._arg'),
          None,
          TD(cmplx, out=('f', 'd', 'g')),
          ),
'negative':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.negative'),
          'PyUFunc_NegativeTypeResolver',
          TD(ints+flts+timedeltaonly, simd=[('avx2', ints)]),
          TD(cmplx, f='neg'),
          TD(O, f='PyNumber_Negative'),
          ),
'positive':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.positive'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(ints+flts+timedeltaonly),
          TD(cmplx, f='pos'),
          TD(O, f='PyNumber_Positive'),
          ),
'sign':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sign'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(nobool_or_datetime),
          ),
'greater':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'greater_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.greater_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'less':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'less_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.less_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'not_equal':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.not_equal'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(all, out='?', simd=[('avx2', ints)]),
          [TypeDescription('O', FullTypeDescr, 'OO', 'O')],
          TD('O', out='?'),
          ),
'logical_and':
    Ufunc(2, 1, True_,
          docstrings.get('numpy.core.umath.logical_and'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalAnd'),
          ),
'logical_not':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.logical_not'),
          None,
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalNot'),
          ),
'logical_or':
    Ufunc(2, 1, False_,
          docstrings.get('numpy.core.umath.logical_or'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?', simd=[('avx2', ints)]),
          TD(O, f='npy_ObjectLogicalOr'),
          ),
'logical_xor':
    Ufunc(2, 1, False_,
          docstrings.get('numpy.core.umath.logical_xor'),
          'PyUFunc_SimpleBinaryComparisonTypeResolver',
          TD(nodatetime_or_obj, out='?'),
          # TODO: using obj.logical_xor() seems pretty much useless:
          TD(P, f='logical_xor'),
          ),
'maximum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.maximum'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(noobj, simd=[('avx512f', 'fd')]),
          TD(O, f='npy_ObjectMax')
          ),
'minimum':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.minimum'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(noobj, simd=[('avx512f', 'fd')]),
          TD(O, f='npy_ObjectMin')
          ),
'clip':
    Ufunc(3, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.clip'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(noobj),
          [TypeDescription('O', 'npy_ObjectClip', 'OOO', 'O')]
          ),
'fmax':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmax'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMax')
          ),
'fmin':
    Ufunc(2, 1, ReorderableNone,
          docstrings.get('numpy.core.umath.fmin'),
          'PyUFunc_SimpleUniformOperationTypeResolver',
          TD(noobj),
          TD(O, f='npy_ObjectMin')
          ),
'logaddexp':
    Ufunc(2, 1, MinusInfinity,
          docstrings.get('numpy.core.umath.logaddexp'),
          None,
          TD(flts, f="logaddexp", astype={'e': 'f'})
          ),
'logaddexp2':
    Ufunc(2, 1, MinusInfinity,
          docstrings.get('numpy.core.umath.logaddexp2'),
          None,
          TD(flts, f="logaddexp2", astype={'e': 'f'})
          ),
'bitwise_and':
    Ufunc(2, 1, AllOnes,
          docstrings.get('numpy.core.umath.bitwise_and'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_And'),
          ),
'bitwise_or':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.bitwise_or'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Or'),
          ),
'bitwise_xor':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.bitwise_xor'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Xor'),
          ),
'invert':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.invert'),
          None,
          TD(bints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Invert'),
          ),
'left_shift':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.left_shift'),
          None,
          TD(ints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Lshift'),
          ),
'right_shift':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.right_shift'),
          None,
          TD(ints, simd=[('avx2', ints)]),
          TD(O, f='PyNumber_Rshift'),
          ),
'heaviside':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.heaviside'),
          None,
          TD(flts, f='heaviside', astype={'e': 'f'}),
          ),
'degrees':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.degrees'),
          None,
          TD(fltsP, f='degrees', astype={'e': 'f'}),
          ),
'rad2deg':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rad2deg'),
          None,
          TD(fltsP, f='rad2deg', astype={'e': 'f'}),
          ),
'radians':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.radians'),
          None,
          TD(fltsP, f='radians', astype={'e': 'f'}),
          ),
'deg2rad':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.deg2rad'),
          None,
          TD(fltsP, f='deg2rad', astype={'e': 'f'}),
          ),
'arccos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccos'),
          None,
          TD('e', f='acos', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='acos', astype={'e': 'f'}),
          TD(P, f='arccos'),
          ),
'arccosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arccosh'),
          None,
          TD('e', f='acosh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='acosh', astype={'e': 'f'}),
          TD(P, f='arccosh'),
          ),
'arcsin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsin'),
          None,
          TD('e', f='asin', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='asin', astype={'e': 'f'}),
          TD(P, f='arcsin'),
          ),
'arcsinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arcsinh'),
          None,
          TD('e', f='asinh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='asinh', astype={'e': 'f'}),
          TD(P, f='arcsinh'),
          ),
'arctan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctan'),
          None,
          TD('e', f='atan', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='atan', astype={'e': 'f'}),
          TD(P, f='arctan'),
          ),
'arctanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.arctanh'),
          None,
          TD('e', f='atanh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='atanh', astype={'e': 'f'}),
          TD(P, f='arctanh'),
          ),
'cos':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cos'),
          None,
          TD('e', f='cos', astype={'e': 'f'}),
          TD('f', dispatch=[('loops_trigonometric', 'f')]),
          TD('d', dispatch=[('loops_umath_fp', 'd')]),
          TD('fdg' + cmplx, f='cos'),
          TD(P, f='cos'),
          ),
'sin':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sin'),
          None,
          TD('e', f='sin', astype={'e': 'f'}),
          TD('f', dispatch=[('loops_trigonometric', 'f')]),
          TD('d', dispatch=[('loops_umath_fp', 'd')]),
          TD('fdg' + cmplx, f='sin'),
          TD(P, f='sin'),
          ),
'tan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tan'),
          None,
          TD('e', f='tan', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='tan', astype={'e': 'f'}),
          TD(P, f='tan'),
          ),
'cosh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cosh'),
          None,
          TD('e', f='cosh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='cosh', astype={'e': 'f'}),
          TD(P, f='cosh'),
          ),
'sinh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sinh'),
          None,
          TD('e', f='sinh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='sinh', astype={'e': 'f'}),
          TD(P, f='sinh'),
          ),
'tanh':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.tanh'),
          None,
          TD('e', f='tanh', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='tanh', astype={'e': 'f'}),
          TD(P, f='tanh'),
          ),
'exp':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp'),
          None,
          TD('e', f='exp', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_exponent_log', 'fd')]),
          TD('fdg' + cmplx, f='exp'),
          TD(P, f='exp'),
          ),
'exp2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.exp2'),
          None,
          TD('e', f='exp2', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='exp2', astype={'e': 'f'}),
          TD(P, f='exp2'),
          ),
'expm1':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.expm1'),
          None,
          TD('e', f='expm1', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='expm1', astype={'e': 'f'}),
          TD(P, f='expm1'),
          ),
'log':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log'),
          None,
          TD('e', f='log', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_exponent_log', 'fd')]),
          TD('fdg' + cmplx, f='log'),
          TD(P, f='log'),
          ),
'log2':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log2'),
          None,
          TD('e', f='log2', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='log2', astype={'e': 'f'}),
          TD(P, f='log2'),
          ),
'log10':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log10'),
          None,
          TD('e', f='log10', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='log10', astype={'e': 'f'}),
          TD(P, f='log10'),
          ),
'log1p':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.log1p'),
          None,
          TD('e', f='log1p', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(inexact, f='log1p', astype={'e': 'f'}),
          TD(P, f='log1p'),
          ),
'sqrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.sqrt'),
          None,
          TD('e', f='sqrt', astype={'e': 'f'}),
          TD(inexactvec, dispatch=[('loops_unary_fp', 'fd')]),
          TD('fdg' + cmplx, f='sqrt'),
          TD(P, f='sqrt'),
          ),
'cbrt':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.cbrt'),
          None,
          TD('e', f='cbrt', astype={'e': 'f'}),
          TD('fd', dispatch=[('loops_umath_fp', 'fd')]),
          TD(flts, f='cbrt', astype={'e': 'f'}),
          TD(P, f='cbrt'),
          ),
'ceil':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.ceil'),
          None,
          TD('e', f='ceil', astype={'e': 'f'}),
          TD(inexactvec, dispatch=[('loops_unary_fp', 'fd')]),
          TD('fdg', f='ceil'),
          TD(O, f='npy_ObjectCeil'),
          ),
'trunc':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.trunc'),
          None,
          TD('e', f='trunc', astype={'e': 'f'}),
          TD(inexactvec, simd=[('fma', 'fd'), ('avx512f', 'fd')]),
          TD('fdg', f='trunc'),
          TD(O, f='npy_ObjectTrunc'),
          ),
'fabs':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.fabs'),
          None,
          TD(flts, f='fabs', astype={'e': 'f'}),
          TD(P, f='fabs'),
       ),
'floor':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.floor'),
          None,
          TD('e', f='floor', astype={'e': 'f'}),
          TD(inexactvec, simd=[('fma', 'fd'), ('avx512f', 'fd')]),
          TD('fdg', f='floor'),
          TD(O, f='npy_ObjectFloor'),
          ),
'rint':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.rint'),
          None,
          TD('e', f='rint', astype={'e': 'f'}),
          TD(inexactvec, simd=[('fma', 'fd'), ('avx512f', 'fd')]),
          TD('fdg' + cmplx, f='rint'),
          TD(P, f='rint'),
          ),
'arctan2':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.arctan2'),
          None,
          TD(flts, f='atan2', astype={'e': 'f'}),
          TD(P, f='arctan2'),
          ),
'remainder':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.remainder'),
          'PyUFunc_RemainderTypeResolver',
          TD(intflt),
          [TypeDescription('m', FullTypeDescr, 'mm', 'm')],
          TD(O, f='PyNumber_Remainder'),
          ),
'divmod':
    Ufunc(2, 2, None,
          docstrings.get('numpy.core.umath.divmod'),
          'PyUFunc_DivmodTypeResolver',
          TD(intflt),
          [TypeDescription('m', FullTypeDescr, 'mm', 'qm')],
          # TD(O, f='PyNumber_Divmod'),  # gh-9730
          ),
'hypot':
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.hypot'),
          None,
          TD(flts, f='hypot', astype={'e': 'f'}),
          TD(P, f='hypot'),
          ),
'isnan':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnan'),
          'PyUFunc_IsFiniteTypeResolver',
          TD(noobj, simd=[('avx512_skx', 'fd')], out='?'),
          ),
'isnat':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isnat'),
          'PyUFunc_IsNaTTypeResolver',
          TD(times, out='?'),
          ),
'isinf':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isinf'),
          'PyUFunc_IsFiniteTypeResolver',
          TD(noobj, simd=[('avx512_skx', 'fd')], out='?'),
          ),
'isfinite':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.isfinite'),
          'PyUFunc_IsFiniteTypeResolver',
          TD(noobj, simd=[('avx512_skx', 'fd')], out='?'),
          ),
'signbit':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.signbit'),
          None,
          TD(flts, simd=[('avx512_skx', 'fd')], out='?'),
          ),
'copysign':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.copysign'),
          None,
          TD(flts),
          ),
'nextafter':
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.nextafter'),
          None,
          TD(flts),
          ),
'spacing':
    Ufunc(1, 1, None,
          docstrings.get('numpy.core.umath.spacing'),
          None,
          TD(flts),
          ),
'modf':
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.modf'),
          None,
          TD(flts),
          ),
'ldexp' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.ldexp'),
          None,
          [TypeDescription('e', None, 'ei', 'e'),
          TypeDescription('f', None, 'fi', 'f', dispatch='loops_exponent_log'),
          TypeDescription('e', FuncNameSuffix('long'), 'el', 'e'),
          TypeDescription('f', FuncNameSuffix('long'), 'fl', 'f'),
          TypeDescription('d', None, 'di', 'd', dispatch='loops_exponent_log'),
          TypeDescription('d', FuncNameSuffix('long'), 'dl', 'd'),
          TypeDescription('g', None, 'gi', 'g'),
          TypeDescription('g', FuncNameSuffix('long'), 'gl', 'g'),
          ],
          ),
'frexp' :
    Ufunc(1, 2, None,
          docstrings.get('numpy.core.umath.frexp'),
          None,
          [TypeDescription('e', None, 'e', 'ei'),
          TypeDescription('f', None, 'f', 'fi', dispatch='loops_exponent_log'),
          TypeDescription('d', None, 'd', 'di', dispatch='loops_exponent_log'),
          TypeDescription('g', None, 'g', 'gi'),
          ],
          ),
'gcd' :
    Ufunc(2, 1, Zero,
          docstrings.get('numpy.core.umath.gcd'),
          "PyUFunc_SimpleUniformOperationTypeResolver",
          TD(ints),
          TD('O', f='npy_ObjectGCD'),
          ),
'lcm' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.lcm'),
          "PyUFunc_SimpleUniformOperationTypeResolver",
          TD(ints),
          TD('O', f='npy_ObjectLCM'),
          ),
'matmul' :
    Ufunc(2, 1, None,
          docstrings.get('numpy.core.umath.matmul'),
          "PyUFunc_SimpleUniformOperationTypeResolver",
          TD(notimes_or_obj),
          TD(O),
          signature='(n?,k),(k,m?)->(n?,m?)',
          ),
}

def indent(st, spaces):
    indentation = ' '*spaces
    indented = indentation + st.replace('\n', '\n'+indentation)
    # trim off any trailing spaces
    indented = re.sub(r' +$', r'', indented)
    return indented

# maps [nin, nout][type] to a suffix
arity_lookup = {
    (1, 1): {
        'e': 'e_e',
        'f': 'f_f',
        'd': 'd_d',
        'g': 'g_g',
        'F': 'F_F',
        'D': 'D_D',
        'G': 'G_G',
        'O': 'O_O',
        'P': 'O_O_method',
    },
    (2, 1): {
        'e': 'ee_e',
        'f': 'ff_f',
        'd': 'dd_d',
        'g': 'gg_g',
        'F': 'FF_F',
        'D': 'DD_D',
        'G': 'GG_G',
        'O': 'OO_O',
        'P': 'OO_O_method',
    },
    (3, 1): {
        'O': 'OOO_O',
    }
}

#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function.

def make_arrays(funcdict):
    # functions array contains an entry for every type implemented NULL
    # should be placed where PyUfunc_ style function will be filled in
    # later
    code1list = []
    code2list = []
    dispdict  = {}
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        funclist = []
        datalist = []
        siglist = []
        k = 0
        sub = 0

        for t in uf.type_descriptions:
            cfunc_alias = t.cfunc_alias if t.cfunc_alias else name
            cfunc_fname = None
            if t.func_data is FullTypeDescr:
                tname = english_upper(chartoname[t.type])
                datalist.append('(void *)NULL')
                cfunc_fname = f"{tname}_{t.in_}_{t.out}_{cfunc_alias}"
            elif isinstance(t.func_data, FuncNameSuffix):
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                cfunc_fname = f"{tname}_{cfunc_alias}_{t.func_data.suffix}"
            elif t.func_data is None:
                datalist.append('(void *)NULL')
                tname = english_upper(chartoname[t.type])
                cfunc_fname = f"{tname}_{cfunc_alias}"
                if t.simd is not None:
                    for vt in t.simd:
                        code2list.append(textwrap.dedent("""\
                        #ifdef HAVE_ATTRIBUTE_TARGET_{ISA}
                        if (NPY_CPU_HAVE({ISA})) {{
                            {fname}_functions[{idx}] = {cname}_{isa};
                        }}
                        #endif
                        """).format(
                            ISA=vt.upper(), isa=vt,
                            fname=name, cname=cfunc_fname, idx=k
                        ))
            else:
                try:
                    thedict = arity_lookup[uf.nin, uf.nout]
                except KeyError as e:
                    raise ValueError(
                        f"Could not handle {name}[{t.type}] "
                        f"with nin={uf.nin}, nout={uf.nout}"
                    ) from None

                astype = ''
                if not t.astype is None:
                    astype = '_As_%s' % thedict[t.astype]
                astr = ('%s_functions[%d] = PyUFunc_%s%s;' %
                           (name, k, thedict[t.type], astype))
                code2list.append(astr)
                if t.type == 'O':
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                elif t.type == 'P':
                    datalist.append('(void *)"%s"' % t.func_data)
                else:
                    astr = ('%s_data[%d] = (void *) %s;' %
                               (name, k, t.func_data))
                    code2list.append(astr)
                    datalist.append('(void *)NULL')
                    #datalist.append('(void *)%s' % t.func_data)
                sub += 1

            if cfunc_fname:
                funclist.append(cfunc_fname)
                if t.dispatch:
                    dispdict.setdefault(t.dispatch, []).append((name, k, cfunc_fname))
            else:
                funclist.append('NULL')

            for x in t.in_ + t.out:
                siglist.append('NPY_%s' % (english_upper(chartoname[x]),))

            k += 1

        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = {%s};"
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = {%s};"
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = {%s};"
                         % (name, signames))

    for dname, funcs in dispdict.items():
        code2list.append(textwrap.dedent(f"""
            #ifndef NPY_DISABLE_OPTIMIZATION
            #include "{dname}.dispatch.h"
            #endif
        """))
        for (ufunc_name, func_idx, cfunc_name) in funcs:
            code2list.append(textwrap.dedent(f"""\
                NPY_CPU_DISPATCH_CALL_XB({ufunc_name}_functions[{func_idx}] = {cfunc_name});
            """))
    return "\n".join(code1list), "\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    names = sorted(funcdict.keys())
    for name in names:
        uf = funcdict[name]
        mlist = []
        docstring = textwrap.dedent(uf.docstring).strip()
        docstring = docstring.encode('unicode-escape').decode('ascii')
        docstring = docstring.replace(r'"', r'\"')
        docstring = docstring.replace(r"'", r"\'")
        # Split the docstring because some compilers (like MS) do not like big
        # string literal in C code. We split at endlines because textwrap.wrap
        # do not play well with \n
        docstring = '\\n\"\"'.join(docstring.split(r"\n"))
        if uf.signature is None:
            sig = "NULL"
        else:
            sig = '"{}"'.format(uf.signature)
        fmt = textwrap.dedent("""\
            identity = {identity_expr};
            if ({has_identity} && identity == NULL) {{
                return -1;
            }}
            f = PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
                {name}_functions, {name}_data, {name}_signatures, {nloops},
                {nin}, {nout}, {identity}, "{name}",
                "{doc}", 0, {sig}, identity
            );
            if ({has_identity}) {{
                Py_DECREF(identity);
            }}
            if (f == NULL) {{
                return -1;
            }}
        """)
        args = dict(
            name=name, nloops=len(uf.type_descriptions),
            nin=uf.nin, nout=uf.nout,
            has_identity='0' if uf.identity is None_ else '1',
            identity='PyUFunc_IdentityValue',
            identity_expr=uf.identity,
            doc=docstring,
            sig=sig,
        )

        # Only PyUFunc_None means don't reorder - we pass this using the old
        # argument
        if uf.identity is None_:
            args['identity'] = 'PyUFunc_None'
            args['identity_expr'] = 'NULL'

        mlist.append(fmt.format(**args))
        if uf.typereso is not None:
            mlist.append(
                r"((PyUFuncObject *)f)->type_resolver = &%s;" % uf.typereso)
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);""" % name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))
    return '\n'.join(code3list)


def make_code(funcdict, filename):
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2, 4)
    code3 = indent(code3, 4)
    code = textwrap.dedent(r"""

    /** Warning this file is autogenerated!!!

        Please make changes to the code generator program (%s)
    **/
    #include "ufunc_object.h"
    #include "ufunc_type_resolution.h"
    #include "loops.h"
    #include "matmul.h"
    #include "clip.h"
    %s

    static int
    InitOperators(PyObject *dictionary) {
        PyObject *f, *identity;

    %s
    %s

        return 0;
    }
    """) % (filename, code1, code2, code3)
    return code


if __name__ == "__main__":
    filename = __file__
    code = make_code(defdict, filename)
    with open('__umath_generated.c', 'w') as fid:
        fid.write(code)
