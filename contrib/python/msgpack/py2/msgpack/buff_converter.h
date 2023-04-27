#include "Python.h"

/* cython does not support this preprocessor check => write it in raw C */
#if PY_MAJOR_VERSION == 2
static PyObject *
buff_to_buff(char *buff, Py_ssize_t size)
{
    return PyBuffer_FromMemory(buff, size);
}

#elif (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 3)
static PyObject *
buff_to_buff(char *buff, Py_ssize_t size)
{
    return PyMemoryView_FromMemory(buff, size, PyBUF_READ);
}
#else
static PyObject *
buff_to_buff(char *buff, Py_ssize_t size)
{
    Py_buffer pybuf;
    if (PyBuffer_FillInfo(&pybuf, NULL, buff, size, 1, PyBUF_FULL_RO) == -1) {
        return NULL;
    }

    return PyMemoryView_FromBuffer(&pybuf);
}
#endif
