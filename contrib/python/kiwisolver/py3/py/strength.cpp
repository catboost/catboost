/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "util.h"


#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-writable-strings"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wwrite-strings"
#endif


namespace kiwisolver
{


namespace
{


void
strength_dealloc( PyObject* self )
{
	Py_TYPE( self )->tp_free( self );
}


PyObject*
strength_weak( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::weak );
}


PyObject*
strength_medium( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::medium );
}


PyObject*
strength_strong( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::strong );
}


PyObject*
strength_required( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::required );
}


PyObject*
strength_create( strength* self, PyObject* args )
{
	PyObject* pya;
	PyObject* pyb;
	PyObject* pyc;
	PyObject* pyw = 0;
	if( !PyArg_ParseTuple( args, "OOO|O", &pya, &pyb, &pyc, &pyw ) )
		return 0;
	double a, b, c;
	double w = 1.0;
	if( !convert_to_double( pya, a ) )
		return 0;
	if( !convert_to_double( pyb, b ) )
		return 0;
	if( !convert_to_double( pyc, c ) )
		return 0;
	if( pyw && !convert_to_double( pyw, w ) )
		return 0;
	return PyFloat_FromDouble( kiwi::strength::create( a, b, c, w ) );
}


static PyGetSetDef
strength_getset[] = {
	{ "weak", ( getter )strength_weak, 0,
	  "The predefined weak strength." },
	{ "medium", ( getter )strength_medium, 0,
	  "The predefined medium strength." },
	{ "strong", ( getter )strength_strong, 0,
	  "The predefined strong strength." },
	{ "required", ( getter )strength_required, 0,
	  "The predefined required strength." },
	{ 0 } // sentinel
};


static PyMethodDef
strength_methods[] = {
	{ "create", ( PyCFunction )strength_create, METH_VARARGS,
	  "Create a strength from constituent values and optional weight." },
	{ 0 } // sentinel
};



static PyType_Slot strength_Type_slots[] = {
    { Py_tp_dealloc, void_cast( strength_dealloc ) },      /* tp_dealloc */
    { Py_tp_getset, void_cast( strength_getset ) },        /* tp_getset */
    { Py_tp_methods, void_cast( strength_methods ) },      /* tp_methods */
    { Py_tp_alloc, void_cast( PyType_GenericAlloc ) },     /* tp_alloc */
    { Py_tp_free, void_cast( PyObject_Del ) },          /* tp_free */
    { 0, 0 },
};


} // namespace


// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject* strength::TypeObject = NULL;


PyType_Spec strength::TypeObject_Spec = {
	"kiwisolver.strength",             /* tp_name */
	sizeof( strength ),                /* tp_basicsize */
	0,                                 /* tp_itemsize */
	Py_TPFLAGS_DEFAULT,                /* tp_flags */
    strength_Type_slots                /* slots */
};


bool strength::Ready()
{
    // The reference will be handled by the module to which we will add the type
	TypeObject = pytype_cast( PyType_FromSpec( &TypeObject_Spec ) );
    if( !TypeObject )
    {
        return false;
    }
    return true;
}


}  // namespace
