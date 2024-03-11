/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "types.h"

#define PY_KIWI_VERSION "1.3.2"

namespace
{


bool ready_types()
{
    using namespace kiwisolver;
    if( !Variable::Ready() )
    {
        return false;
    }
    if( !Term::Ready() )
    {
        return false;
    }
    if( !Expression::Ready() )
    {
        return false;
    }
    if( !Constraint::Ready() )
    {
        return false;
    }
    if( !strength::Ready() )
    {
        return false;
    }
    if( !Solver::Ready() )
    {
        return false;
    }
    return true;
}

bool add_objects( PyObject* mod )
{
	using namespace kiwisolver;

    cppy::ptr kiwiversion( PyUnicode_FromString( KIWI_VERSION ) );
    if( !kiwiversion )
    {
        return false;
    }
    cppy::ptr pyversion( PyUnicode_FromString( PY_KIWI_VERSION ) );
    if( !pyversion )
    {
        return false;
    }
    cppy::ptr pystrength( PyType_GenericNew( strength::TypeObject, 0, 0 ) );
    if( !pystrength )
    {
        return false;
    }

    if( PyModule_AddObject( mod, "__version__", pyversion.get() ) < 0 )
    {
		return false;
	}
    pyversion.release();

    if( PyModule_AddObject( mod, "__kiwi_version__", kiwiversion.get() ) < 0 )
    {
		return false;
	}
    kiwiversion.release();

    if( PyModule_AddObject( mod, "strength", pystrength.get() ) < 0 )
    {
		return false;
	}
    pystrength.release();

    // Variable
    cppy::ptr var( pyobject_cast( Variable::TypeObject ) );
	if( PyModule_AddObject( mod, "Variable", var.get() ) < 0 )
	{
		return false;
	}
    var.release();

    // Term
    cppy::ptr term( pyobject_cast( Term::TypeObject ) );
	if( PyModule_AddObject( mod, "Term", term.get() ) < 0 )
	{
		return false;
	}
    term.release();

    // Expression
    cppy::ptr expr( pyobject_cast( Expression::TypeObject ) );
	if( PyModule_AddObject( mod, "Expression", expr.get() ) < 0 )
	{
		return false;
	}
    expr.release();

    // Constraint
    cppy::ptr cons( pyobject_cast( Constraint::TypeObject ) );
	if( PyModule_AddObject( mod, "Constraint", cons.get() ) < 0 )
	{
		return false;
	}
    cons.release();

    cppy::ptr solver( pyobject_cast( Solver::TypeObject ) );
	if( PyModule_AddObject( mod, "Solver", solver.get() ) < 0 )
	{
		return false;
	}
    solver.release();

    PyModule_AddObject( mod, "DuplicateConstraint", DuplicateConstraint );
    PyModule_AddObject( mod, "UnsatisfiableConstraint", UnsatisfiableConstraint );
    PyModule_AddObject( mod, "UnknownConstraint", UnknownConstraint );
    PyModule_AddObject( mod, "DuplicateEditVariable", DuplicateEditVariable );
    PyModule_AddObject( mod, "UnknownEditVariable", UnknownEditVariable );
    PyModule_AddObject( mod, "BadRequiredStrength", BadRequiredStrength );

	return true;
}


int
catom_modexec( PyObject *mod )
{
    if( !ready_types() )
    {
        return -1;
    }
    if( !kiwisolver::init_exceptions() )
    {
        return -1;
    }
    if( !add_objects( mod ) )
    {
        return -1;
    }


    return 0;
}


static PyMethodDef
kiwisolver_methods[] = {
    { 0 } // Sentinel
};


PyModuleDef_Slot kiwisolver_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>( catom_modexec ) },
    {0, NULL}
};


struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "kiwisolver",
        "kiwisolver extension module",
        0,
        kiwisolver_methods,
        kiwisolver_slots,
        NULL,
        NULL,
        NULL
};

}  // namespace


PyMODINIT_FUNC PyInit_kiwisolver( void )
{
    return PyModuleDef_Init( &moduledef );
}
