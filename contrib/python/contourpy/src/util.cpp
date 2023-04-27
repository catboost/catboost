#include "util.h"
#include <thread>

namespace contourpy {

index_t Util::get_max_threads()
{
    return static_cast<index_t>(std::thread::hardware_concurrency());
}

} // namespace contourpy
