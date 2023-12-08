/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2019, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#include <algorithm>
#include <sstream>
#include <cppy/cppy.h>
#include <kiwi/kiwi.h>
#include "types.h"
#include "util.h"

namespace kiwisolver
{

namespace
{

PyObject *
Constraint_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"expression", "op", "strength", 0};
    PyObject *pyexpr;
    PyObject *pyop;
    PyObject *pystrength = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO|O:__new__", const_cast<char **>(kwlist),
            &pyexpr, &pyop, &pystrength))
        return 0;
    if (!Expression::TypeCheck(pyexpr))
        return cppy::type_error(pyexpr, "Expression");
    kiwi::RelationalOperator op;
    if (!convert_to_relational_op(pyop, op))
        return 0;
    double strength = kiwi::strength::required;
    if (pystrength && !convert_to_strength(pystrength, strength))
        return 0;
    cppy::ptr pycn(PyType_GenericNew(type, args, kwargs));
    if (!pycn)
        return 0;
    Constraint *cn = reinterpret_cast<Constraint *>(pycn.get());
    cn->expression = reduce_expression(pyexpr);
    if (!cn->expression)
        return 0;
    kiwi::Expression expr(convert_to_kiwi_expression(cn->expression));
    new (&cn->constraint) kiwi::Constraint(expr, op, strength);
    return pycn.release();
}

void Constraint_clear(Constraint *self)
{
    Py_CLEAR(self->expression);
}

int Constraint_traverse(Constraint *self, visitproc visit, void *arg)
{
    Py_VISIT(self->expression);
#if PY_VERSION_HEX >= 0x03090000
    // This was not needed before Python 3.9 (Python issue 35810 and 40217)
    Py_VISIT(Py_TYPE(self));
#endif
    return 0;
}

void Constraint_dealloc(Constraint *self)
{
    PyObject_GC_UnTrack(self);
    Constraint_clear(self);
    self->constraint.~Constraint();
    Py_TYPE(self)->tp_free(pyobject_cast(self));
}

PyObject *
Constraint_repr(Constraint *self)
{
    std::stringstream stream;
    Expression *expr = reinterpret_cast<Expression *>(self->expression);
    Py_ssize_t size = PyTuple_GET_SIZE(expr->terms);
    for (Py_ssize_t i = 0; i < size; ++i)
    {
        PyObject *item = PyTuple_GET_ITEM(expr->terms, i);
        Term *term = reinterpret_cast<Term *>(item);
        stream << term->coefficient << " * ";
        stream << reinterpret_cast<Variable *>(term->variable)->variable.name();
        stream << " + ";
    }
    stream << expr->constant;
    switch (self->constraint.op())
    {
    case kiwi::OP_EQ:
        stream << " == 0";
        break;
    case kiwi::OP_LE:
        stream << " <= 0";
        break;
    case kiwi::OP_GE:
        stream << " >= 0";
        break;
    }
    stream << " | strength = " << self->constraint.strength();
    return PyUnicode_FromString(stream.str().c_str());
}

PyObject *
Constraint_expression(Constraint *self)
{
    return cppy::incref(self->expression);
}

PyObject *
Constraint_op(Constraint *self)
{
    PyObject *res = 0;
    switch (self->constraint.op())
    {
    case kiwi::OP_EQ:
        res = PyUnicode_FromString("==");
        break;
    case kiwi::OP_LE:
        res = PyUnicode_FromString("<=");
        break;
    case kiwi::OP_GE:
        res = PyUnicode_FromString(">=");
        break;
    }
    return res;
}

PyObject *
Constraint_strength(Constraint *self)
{
    return PyFloat_FromDouble(self->constraint.strength());
}

PyObject *
Constraint_or(PyObject *pyoldcn, PyObject *value)
{
    if (!Constraint::TypeCheck(pyoldcn))
        std::swap(pyoldcn, value);
    double strength;
    if (!convert_to_strength(value, strength))
        return 0;
    PyObject *pynewcn = PyType_GenericNew(Constraint::TypeObject, 0, 0);
    if (!pynewcn)
        return 0;
    Constraint *oldcn = reinterpret_cast<Constraint *>(pyoldcn);
    Constraint *newcn = reinterpret_cast<Constraint *>(pynewcn);
    newcn->expression = cppy::incref(oldcn->expression);
    new (&newcn->constraint) kiwi::Constraint(oldcn->constraint, strength);
    return pynewcn;
}

static PyMethodDef
    Constraint_methods[] = {
        {"expression", (PyCFunction)Constraint_expression, METH_NOARGS,
         "Get the expression object for the constraint."},
        {"op", (PyCFunction)Constraint_op, METH_NOARGS,
         "Get the relational operator for the constraint."},
        {"strength", (PyCFunction)Constraint_strength, METH_NOARGS,
         "Get the strength for the constraint."},
        {0} // sentinel
};

static PyType_Slot Constraint_Type_slots[] = {
    {Py_tp_dealloc, void_cast(Constraint_dealloc)},   /* tp_dealloc */
    {Py_tp_traverse, void_cast(Constraint_traverse)}, /* tp_traverse */
    {Py_tp_clear, void_cast(Constraint_clear)},       /* tp_clear */
    {Py_tp_repr, void_cast(Constraint_repr)},         /* tp_repr */
    {Py_tp_methods, void_cast(Constraint_methods)},   /* tp_methods */
    {Py_tp_new, void_cast(Constraint_new)},           /* tp_new */
    {Py_tp_alloc, void_cast(PyType_GenericAlloc)},    /* tp_alloc */
    {Py_tp_free, void_cast(PyObject_GC_Del)},         /* tp_free */
    {Py_nb_or, void_cast(Constraint_or)},             /* nb_or */
    {0, 0},
};

} // namespace

// Initialize static variables (otherwise the compiler eliminates them)
PyTypeObject *Constraint::TypeObject = NULL;

PyType_Spec Constraint::TypeObject_Spec = {
    "kiwisolver.Constraint", /* tp_name */
    sizeof(Constraint),      /* tp_basicsize */
    0,                       /* tp_itemsize */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_BASETYPE, /* tp_flags */
    Constraint_Type_slots    /* slots */
};

bool Constraint::Ready()
{
    // The reference will be handled by the module to which we will add the type
    TypeObject = pytype_cast(PyType_FromSpec(&TypeObject_Spec));
    if (!TypeObject)
    {
        return false;
    }
    return true;
}

} // namespace kiwisolver
