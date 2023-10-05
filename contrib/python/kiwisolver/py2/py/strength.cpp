/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#include <Python.h>
#include <kiwi/kiwi.h>
#include "pythonhelpers.h"
#include "util.h"


using namespace PythonHelpers;


struct strength
{
	PyObject_HEAD;
};


static void
strength_dealloc( PyObject* self )
{
	Py_TYPE( self )->tp_free( self );
}


static PyObject*
strength_weak( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::weak );
}


static PyObject*
strength_medium( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::medium );
}


static PyObject*
strength_strong( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::strong );
}


static PyObject*
strength_required( strength* self )
{
	return PyFloat_FromDouble( kiwi::strength::required );
}


static PyObject*
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


PyTypeObject strength_Type = {
	PyVarObject_HEAD_INIT( &PyType_Type, 0 )
	"kiwisolver.strength",                  /* tp_name */
	sizeof( strength ),                     /* tp_basicsize */
	0,                                      /* tp_itemsize */
	(destructor)strength_dealloc,           /* tp_dealloc */
	(printfunc)0,                           /* tp_print */
	(getattrfunc)0,                         /* tp_getattr */
	(setattrfunc)0,                         /* tp_setattr */
#if PY_VERSION_HEX >= 0x03050000
	( PyAsyncMethods* )0,                   /* tp_as_async */
#elif PY_VERSION_HEX >= 0x03000000
	( void* ) 0,                            /* tp_reserved */
#else
	( cmpfunc )0,                           /* tp_compare */
#endif
	(reprfunc)0,                            /* tp_repr */
	(PyNumberMethods*)0,                    /* tp_as_number */
	(PySequenceMethods*)0,                  /* tp_as_sequence */
	(PyMappingMethods*)0,                   /* tp_as_mapping */
	(hashfunc)0,                            /* tp_hash */
	(ternaryfunc)0,                         /* tp_call */
	(reprfunc)0,                            /* tp_str */
	(getattrofunc)0,                        /* tp_getattro */
	(setattrofunc)0,                        /* tp_setattro */
	(PyBufferProcs*)0,                      /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,                     /* tp_flags */
	0,                                      /* Documentation string */
	(traverseproc)0,                        /* tp_traverse */
	(inquiry)0,                             /* tp_clear */
	(richcmpfunc)0,                         /* tp_richcompare */
	0,                                      /* tp_weaklistoffset */
	(getiterfunc)0,                         /* tp_iter */
	(iternextfunc)0,                        /* tp_iternext */
	(struct PyMethodDef*)strength_methods,  /* tp_methods */
	(struct PyMemberDef*)0,                 /* tp_members */
	strength_getset,                        /* tp_getset */
	0,                                      /* tp_base */
	0,                                      /* tp_dict */
	(descrgetfunc)0,                        /* tp_descr_get */
	(descrsetfunc)0,                        /* tp_descr_set */
	0,                                      /* tp_dictoffset */
	(initproc)0,                            /* tp_init */
	(allocfunc)PyType_GenericAlloc,         /* tp_alloc */
	(newfunc)0,                             /* tp_new */
	(freefunc)PyObject_Del,                 /* tp_free */
	(inquiry)0,                             /* tp_is_gc */
	0,                                      /* tp_bases */
	0,                                      /* tp_mro */
	0,                                      /* tp_cache */
	0,                                      /* tp_subclasses */
	0,                                      /* tp_weaklist */
	(destructor)0                           /* tp_del */
};


int import_strength()
{
	return PyType_Ready( &strength_Type );
}
