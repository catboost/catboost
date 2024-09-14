#ifndef CONTOURPY_CONTOUR_GENERATOR_H
#define CONTOURPY_CONTOUR_GENERATOR_H

#include "common.h"

namespace contourpy {

class ContourGenerator
{
public:
    // Non-copyable and non-moveable.
    ContourGenerator(const ContourGenerator& other) = delete;
    ContourGenerator(const ContourGenerator&& other) = delete;
    ContourGenerator& operator=(const ContourGenerator& other) = delete;
    ContourGenerator& operator=(const ContourGenerator&& other) = delete;

    virtual ~ContourGenerator() = default;

    virtual py::tuple filled(double lower_level, double upper_level) = 0;

    virtual py::sequence lines(double level) = 0;

    virtual py::list multi_filled(const LevelArray levels);
    virtual py::list multi_lines(const LevelArray levels);

protected:
    ContourGenerator() = default;

    // Check levels are acceptable, throw exception if not.
    void check_levels(const LevelArray& levels, bool filled) const;
    void check_levels(double lower_level, double upper_level) const;
};

} // namespace contourpy

#endif // CONTOURPY_CONTOUR_GENERATOR_H
