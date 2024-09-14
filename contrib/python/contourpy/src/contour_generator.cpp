#include "contour_generator.h"
#include "util.h"

namespace contourpy {

void ContourGenerator::check_levels(const LevelArray& levels, bool filled) const
{
    if (levels.ndim() != 1) {
        throw std::domain_error(
            "Levels array must be 1D not " + std::to_string(levels.ndim()) + "D");
    }

    if (filled) {
        auto n = levels.size();
        if (n < 2) {
            throw std::invalid_argument(
                "Levels array must have at least 2 elements, not " + std::to_string(n));
        }

        auto levels_proxy = levels.unchecked<1>();

        // Check levels are not NaN.
        for (decltype(n) i = 0; i < n; i++) {
            if (Util::is_nan(levels_proxy[i]))
                throw std::invalid_argument("Levels must not contain any NaN");
        }

        // Check levels are increasing.
        auto lower_level = levels_proxy[0];
        for (decltype(n) i = 0; i < n-1; i++) {
            auto upper_level = levels_proxy[i+1];
            if (lower_level >= upper_level)
                throw std::invalid_argument("Levels must be increasing");
            lower_level = upper_level;
        }
    }
}

void ContourGenerator::check_levels(double lower_level, double upper_level) const
{
    if (Util::is_nan(lower_level) || Util::is_nan(upper_level))
        throw std::invalid_argument("lower_level and upper_level cannot be NaN");
    if (lower_level >= upper_level)
        throw std::invalid_argument("upper_level must be larger than lower_level");
}

py::list ContourGenerator::multi_filled(const LevelArray levels)
{
    check_levels(levels, true);

    auto levels_proxy = levels.unchecked<1>();
    auto n = levels_proxy.size();

    py::list ret(n-1);
    auto lower_level = levels_proxy[0];
    for (decltype(n) i = 0; i < n-1; i++) {
        auto upper_level = levels_proxy[i+1];
        ret[i] = filled(lower_level, upper_level);
        lower_level = upper_level;
    }

    return ret;
}

py::list ContourGenerator::multi_lines(const LevelArray levels)
{
    check_levels(levels, false);

    auto levels_proxy = levels.unchecked<1>();
    auto n = levels_proxy.size();

    py::list ret(n);
    for (decltype(n) i = 0; i < n; i++)
        ret[i] = lines(levels_proxy[i]);

    return ret;
}

} // namespace contourpy
