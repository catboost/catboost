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

protected:
    ContourGenerator() = default;
};

} // namespace contourpy

#endif // CONTOURPY_CONTOUR_GENERATOR_H
