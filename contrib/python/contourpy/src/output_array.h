#ifndef CONTOURPY_OUTPUT_ARRAY_H
#define CONTOURPY_OUTPUT_ARRAY_H

#include "common.h"
#include <vector>

namespace contourpy {

// A reusable array that is output from C++ to Python.  Depending on the chosen line or fill type,
// it can either be created as a NumPy array that will be directly returned to the Python caller,
// or as a C++ vector that will be further manipulated (such as split up) before being converted to
// NumPy array(s) for returning.  BaseContourGenerator's marching does not care which form it is as
// it just writes values to either array using an incrementing pointer.
template <typename T>
class OutputArray
{
public:
    OutputArray()
        : size(0), start(nullptr), current(nullptr)
    {}

    void clear()
    {
        vector.clear();
        size = 0;
        start = current = nullptr;
    }

    void create_cpp(count_t new_size)
    {
        assert(new_size > 0);
        size = new_size;
        vector.resize(size);
        start = current = vector.data();
    }

    py::array_t<T> create_python(count_t new_size)
    {
        assert(new_size > 0);
        size = new_size;
        py::array_t<T> py_array(size);
        start = current = py_array.mutable_data();
        return py_array;
    }

    py::array_t<T> create_python(count_t shape0, count_t shape1)
    {
        assert(shape0 > 0 && shape1 > 0);
        size = shape0*shape1;
        py::array_t<T> py_array({shape0, shape1});
        start = current = py_array.mutable_data();
        return py_array;
    }

    // Non-copyable and non-moveable.
    OutputArray(const OutputArray& other) = delete;
    OutputArray(const OutputArray&& other) = delete;
    OutputArray& operator=(const OutputArray& other) = delete;
    OutputArray& operator=(const OutputArray&& other) = delete;


    std::vector<T> vector;
    count_t size;
    T* start;               // Start of array, whether C++ or Python.
    T* current;             // Where to write next value to before incrementing.
};

} // namespace contourpy

#endif // CONTOURPY_OUTPUT_ARRAY_H
