#include "util.h"
#include <thread>

namespace contourpy {

bool Util::_nan_loaded = false;

double Util::nan = 0.0;

void Util::ensure_nan_loaded()
{
    if (!_nan_loaded) {
        auto numpy = py::module_::import("numpy");
        nan = numpy.attr("nan").cast<double>();
        _nan_loaded = true;
    }
}

index_t Util::get_max_threads()
{
    return static_cast<index_t>(std::thread::hardware_concurrency());
}

} // namespace contourpy
