//! Low-Level binding for [UFunc API](https://numpy.org/doc/stable/reference/c-api/ufunc.html)

use std::os::raw::*;

use pyo3::{ffi::PyObject, sync::GILOnceCell};

use crate::npyffi::*;

fn mod_name(py: Python<'_>) -> PyResult<&'static str> {
    static MOD_NAME: GILOnceCell<String> = GILOnceCell::new();
    MOD_NAME
        .get_or_try_init(py, || {
            let numpy_core = super::array::numpy_core_name(py)?;
            Ok(format!("{}.umath", numpy_core))
        })
        .map(String::as_str)
}

const CAPSULE_NAME: &str = "_UFUNC_API";

/// A global variable which stores a ['capsule'](https://docs.python.org/3/c-api/capsule.html)
/// pointer to [Numpy UFunc API](https://numpy.org/doc/stable/reference/c-api/ufunc.html).
pub static PY_UFUNC_API: PyUFuncAPI = PyUFuncAPI(GILOnceCell::new());

pub struct PyUFuncAPI(GILOnceCell<*const *const c_void>);

unsafe impl Send for PyUFuncAPI {}

unsafe impl Sync for PyUFuncAPI {}

impl PyUFuncAPI {
    unsafe fn get<'py>(&self, py: Python<'py>, offset: isize) -> *const *const c_void {
        let api = self
            .0
            .get_or_try_init(py, || get_numpy_api(py, mod_name(py)?, CAPSULE_NAME))
            .expect("Failed to access NumPy ufunc API capsule");

        api.offset(offset)
    }
}

impl PyUFuncAPI {
    impl_api![1; PyUFunc_FromFuncAndData(func: *mut PyUFuncGenericFunction, data: *mut *mut c_void, types: *mut c_char, ntypes: c_int, nin: c_int, nout: c_int, identity: c_int, name: *const c_char, doc: *const c_char, unused: c_int) -> *mut PyObject];
    impl_api![2; PyUFunc_RegisterLoopForType(ufunc: *mut PyUFuncObject, usertype: c_int, function: PyUFuncGenericFunction, arg_types: *mut c_int, data: *mut c_void) -> c_int];
    impl_api![3; PyUFunc_GenericFunction(ufunc: *mut PyUFuncObject, args: *mut PyObject, kwds: *mut PyObject, op: *mut *mut PyArrayObject) -> c_int];
    impl_api![4; PyUFunc_f_f_As_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![5; PyUFunc_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![6; PyUFunc_f_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![7; PyUFunc_g_g(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![8; PyUFunc_F_F_As_D_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![9; PyUFunc_F_F(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![10; PyUFunc_D_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![11; PyUFunc_G_G(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![12; PyUFunc_O_O(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![13; PyUFunc_ff_f_As_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![14; PyUFunc_ff_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![15; PyUFunc_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![16; PyUFunc_gg_g(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![17; PyUFunc_FF_F_As_DD_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![18; PyUFunc_DD_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![19; PyUFunc_FF_F(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![20; PyUFunc_GG_G(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![21; PyUFunc_OO_O(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![22; PyUFunc_O_O_method(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![23; PyUFunc_OO_O_method(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![24; PyUFunc_On_Om(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![25; PyUFunc_GetPyValues(name: *mut c_char, bufsize: *mut c_int, errmask: *mut c_int, errobj: *mut *mut PyObject) -> c_int];
    impl_api![26; PyUFunc_checkfperr(errmask: c_int, errobj: *mut PyObject, first: *mut c_int) -> c_int];
    impl_api![27; PyUFunc_clearfperr()];
    impl_api![28; PyUFunc_getfperr() -> c_int];
    impl_api![29; PyUFunc_handlefperr(errmask: c_int, errobj: *mut PyObject, retstatus: c_int, first: *mut c_int) -> c_int];
    impl_api![30; PyUFunc_ReplaceLoopBySignature(func: *mut PyUFuncObject, newfunc: PyUFuncGenericFunction, signature: *mut c_int, oldfunc: *mut PyUFuncGenericFunction) -> c_int];
    impl_api![31; PyUFunc_FromFuncAndDataAndSignature(func: *mut PyUFuncGenericFunction, data: *mut *mut c_void, types: *mut c_char, ntypes: c_int, nin: c_int, nout: c_int, identity: c_int, name: *const c_char, doc: *const c_char, unused: c_int, signature: *const c_char) -> *mut PyObject];
    impl_api![32; PyUFunc_SetUsesArraysAsData(data: *mut *mut c_void, i: usize) -> c_int];
    impl_api![33; PyUFunc_e_e(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![34; PyUFunc_e_e_As_f_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![35; PyUFunc_e_e_As_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![36; PyUFunc_ee_e(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![37; PyUFunc_ee_e_As_ff_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![38; PyUFunc_ee_e_As_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
    impl_api![39; PyUFunc_DefaultTypeResolver(ufunc: *mut PyUFuncObject, casting: NPY_CASTING, operands: *mut *mut PyArrayObject, type_tup: *mut PyObject, out_dtypes: *mut *mut PyArray_Descr) -> c_int];
    impl_api![40; PyUFunc_ValidateCasting(ufunc: *mut PyUFuncObject, casting: NPY_CASTING, operands: *mut *mut PyArrayObject, dtypes: *mut *mut PyArray_Descr) -> c_int];
    impl_api![41; PyUFunc_RegisterLoopForDescr(ufunc: *mut PyUFuncObject, user_dtype: *mut PyArray_Descr, function: PyUFuncGenericFunction, arg_dtypes: *mut *mut PyArray_Descr, data: *mut c_void) -> c_int];
    impl_api![42; PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
        ufunc: *mut PyUFuncObject,
        data: *mut *mut c_void,
        types: *mut c_char,
        ntypes: c_int,
        nin: c_int,
        nout: c_int,
        identity: c_int,
        name: *const c_char,
        doc: *const c_char,
        unused: c_int,
        signature: *const c_char,
        identity_value: *const c_char,
    ) -> c_int];
}
