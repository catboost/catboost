#include "dump.h"

#include "encoder.h"

namespace NSJson {
    void DumpToStream(PyObject* obj, PyObject* stream) {
        try {
            Encode(obj, stream);
        } catch (TValueError& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (yexception& e) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
            }
        } catch (...) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Unexpected error");
            }
        }
    }
}
