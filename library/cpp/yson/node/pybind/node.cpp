#include "node.h"

#include <library/cpp/yson/node/node.h>

#include <library/cpp/pybind/cast.h>

#include <Python.h>

namespace NYT {

    PyObject* BuildPyObject(const TNode& node) {
        switch (node.GetType()) {
            case TNode::Bool:
                return NPyBind::BuildPyObject(node.AsBool());
            case TNode::Int64:
                return NPyBind::BuildPyObject(node.AsInt64());
            case TNode::Uint64:
                return NPyBind::BuildPyObject(node.AsUint64());
            case TNode::Double:
                return NPyBind::BuildPyObject(node.AsDouble());
            case TNode::String:
                return NPyBind::BuildPyObject(node.AsString());
            case TNode::List:
                return NPyBind::BuildPyObject(node.AsList());
            case TNode::Map:
                return NPyBind::BuildPyObject(node.AsMap());
            case TNode::Null:
                Py_RETURN_NONE;
            case TNode::Undefined:
                ythrow TNode::TTypeError() << "BuildPyObject called for undefined TNode";
        }
    }

} // namespace NYT

namespace NPyBind {

    template <>
    bool FromPyObject(PyObject* obj, NYT::TNode& res) {
        if (obj == Py_None) {
            res = NYT::TNode::CreateEntity();
            return true;
        }
        if (PyBool_Check(obj)) {
            res = false;
            return FromPyObject(obj, res.As<bool>());
        }
        if (PyFloat_Check(obj)) {
            res = 0.0;
            return FromPyObject(obj, res.As<double>());
        }
        if (PyString_Check(obj)) {
            res = TString();
            return FromPyObject(obj, res.As<TString>());
        }
        if (PyList_Check(obj)) {
            res = NYT::TNode::CreateList();
            return FromPyObject(obj, res.AsList());
        }
        if (PyDict_Check(obj)) {
            res = NYT::TNode::CreateMap();
            return FromPyObject(obj, res.AsMap());
        }
        if (PyInt_Check(obj)) {
            auto valAsLong = PyInt_AsLong(obj);
            if (valAsLong == -1 && PyErr_Occurred()) {
                return false;
            }
            res = valAsLong;
            return true;
        }
        if (PyLong_Check(obj)) {
            int overflow = 0;
            auto valAsLong = PyLong_AsLongAndOverflow(obj, &overflow);
            if (!overflow) {
                if (valAsLong == -1 && PyErr_Occurred()) {
                    return false;
                }
                res = valAsLong;
                return true;
            }
            auto valAsULong = PyLong_AsUnsignedLong(obj);
            if (valAsULong == static_cast<decltype(valAsULong)>(-1) && PyErr_Occurred()) {
                return false;
            }
            res = valAsULong;
            return true;
        }
        return false;
    }

} // namespace NPyBind
