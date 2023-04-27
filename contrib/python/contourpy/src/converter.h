#ifndef CONTOURPY_CONVERTER_H
#define CONTOURPY_CONVERTER_H

#include "common.h"

namespace contourpy {

// Conversion of C++ point/code/offset array to return to Python as a NumPy array.
// There are two versions of each function, the first creates, populates and returns a NumPy array
// whereas the second populates one that has already been created. The former are used in serial
// code and the latter in threaded code where the creation and manipulation of NumPy arrays needs to
// be threadlocked whereas the population of those arrays does not.
class Converter
{
public:
    // Create and populate codes array,
    static CodeArray convert_codes(
        count_t point_count, count_t cut_count, const offset_t* cut_start, offset_t subtract);

    // Populate codes array that has already been created.
    static void convert_codes(
        count_t point_count, count_t cut_count, const offset_t* cut_start, offset_t subtract,
        CodeArray::value_type* codes);

    // Create and populate codes array,
    static CodeArray convert_codes_check_closed(
        count_t point_count, count_t cut_count, const offset_t* cut_start, const double* points);

    // Populate codes array that has already been created.
    static void convert_codes_check_closed(
        count_t point_count, count_t cut_count, const offset_t* cut_start, const double* points,
        CodeArray::value_type* codes);

    // Create and populate codes array (single line loop/strip).
    static CodeArray convert_codes_check_closed_single(
        count_t point_count, const double* points);

    // Populate codes array that has already been created (single line loop/strip).
    static void convert_codes_check_closed_single(
        count_t point_count, const double* points, CodeArray::value_type* codes);

    // Create and populate offsets array,
    static OffsetArray convert_offsets(
        count_t offset_count, const offset_t* start, offset_t subtract);

    // Populate offsets array that has already been created.
    static void convert_offsets(
        count_t offset_count, const offset_t* start, offset_t subtract,
        OffsetArray::value_type* offsets);

    // Create and populate points array,
    static PointArray convert_points(count_t point_count, const double* start);

    // Populate points array that has already been created.
    static void convert_points(count_t point_count, const double* start, double* points);

private:
    static void check_max_offset(count_t max_offset);
};

} // namespace contourpy

#endif // CONTOURPY_CONVERTER_H
