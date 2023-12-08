/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <Python.h>
#include <kiwi/kiwi.h>


namespace kiwisolver
{

extern PyObject* DuplicateConstraint;

extern PyObject* UnsatisfiableConstraint;

extern PyObject* UnknownConstraint;

extern PyObject* DuplicateEditVariable;

extern PyObject* UnknownEditVariable;

extern PyObject* BadRequiredStrength;


struct strength
{
	PyObject_HEAD;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();
};


struct Variable
{
	PyObject_HEAD
	PyObject* context;
	kiwi::Variable variable;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();

	static bool TypeCheck( PyObject* obj )
	{
		return PyObject_TypeCheck( obj, TypeObject ) != 0;
	}
};


struct Term
{
	PyObject_HEAD
	PyObject* variable;
	double coefficient;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();

	static bool TypeCheck( PyObject* obj )
	{
		return PyObject_TypeCheck( obj, TypeObject ) != 0;
	}
};


struct Expression
{
	PyObject_HEAD
	PyObject* terms;
	double constant;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();

	static bool TypeCheck( PyObject* obj )
	{
		return PyObject_TypeCheck( obj, TypeObject ) != 0;
	}
};


struct Constraint
{
	PyObject_HEAD
	PyObject* expression;
	kiwi::Constraint constraint;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();

	static bool TypeCheck( PyObject* obj )
	{
		return PyObject_TypeCheck( obj, TypeObject ) != 0;
	}
};


struct Solver
{
	PyObject_HEAD
	kiwi::Solver solver;

    static PyType_Spec TypeObject_Spec;

    static PyTypeObject* TypeObject;

	static bool Ready();

	static bool TypeCheck( PyObject* obj )
	{
		return PyObject_TypeCheck( obj, TypeObject ) != 0;
	}
};


bool init_exceptions();


}  // namespace kiwisolver
