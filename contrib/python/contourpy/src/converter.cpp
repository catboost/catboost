#include "converter.h"
#include "mpl_kind_code.h"
#include <limits>

namespace contourpy {

void Converter::check_max_offset(count_t max_offset)
{
    if (max_offset > std::numeric_limits<OffsetArray::value_type>::max())
        throw std::range_error("Max offset too large to fit in np.uint32. Use smaller chunks.");
}

CodeArray Converter::convert_codes(
    count_t point_count, count_t cut_count, const offset_t* cut_start, offset_t subtract)
{
    assert(point_count > 0 && cut_count > 0 && subtract >= 0);
    assert(cut_start != nullptr);

    index_t codes_shape = static_cast<index_t>(point_count);
    CodeArray py_codes(codes_shape);
    convert_codes(point_count, cut_count, cut_start, subtract, py_codes.mutable_data());
    return py_codes;
}

void Converter::convert_codes(
    count_t point_count, count_t cut_count, const offset_t* cut_start, offset_t subtract,
    CodeArray::value_type* codes)
{
    assert(point_count > 0 && cut_count > 0 && subtract >= 0);
    assert(cut_start != nullptr);
    assert(codes != nullptr);

    std::fill(codes + 1, codes + point_count - 1, LINETO);
    for (decltype(cut_count) i = 0; i < cut_count-1; ++i) {
        codes[cut_start[i] - subtract] = MOVETO;
        codes[cut_start[i+1] - 1 - subtract] = CLOSEPOLY;
    }
}

CodeArray Converter::convert_codes_check_closed(
    count_t point_count, count_t cut_count, const offset_t* cut_start, const double* points)
{
    assert(point_count > 0 && cut_count > 0);
    assert(cut_start != nullptr);
    assert(points != nullptr);

    index_t codes_shape = static_cast<index_t>(point_count);
    CodeArray codes(codes_shape);
    convert_codes_check_closed(point_count, cut_count, cut_start, points, codes.mutable_data());
    return codes;
}

void Converter::convert_codes_check_closed(
    count_t point_count, count_t cut_count, const offset_t* cut_start, const double* points,
    CodeArray::value_type* codes)
{
    assert(point_count > 0 && cut_count > 0);
    assert(cut_start != nullptr);
    assert(points != nullptr);
    assert(codes != nullptr);

    std::fill(codes + 1, codes + point_count, LINETO);
    for (decltype(cut_count) i = 0; i < cut_count-1; ++i) {
        auto start = cut_start[i];
        auto end = cut_start[i+1];
        codes[start] = MOVETO;
        bool closed = points[2*start] == points[2*end-2] &&
                      points[2*start+1] == points[2*end-1];
        if (closed)
            codes[end-1] = CLOSEPOLY;
    }
}

CodeArray Converter::convert_codes_check_closed_single(
    count_t point_count, const double* points)
{
    assert(point_count > 0);
    assert(points != nullptr);

    index_t codes_shape = static_cast<index_t>(point_count);
    CodeArray py_codes(codes_shape);
    convert_codes_check_closed_single(point_count, points, py_codes.mutable_data());
    return py_codes;
}

void Converter::convert_codes_check_closed_single(
    count_t point_count, const double* points, CodeArray::value_type* codes)
{
    assert(point_count > 0);
    assert(points != nullptr);
    assert(codes != nullptr);

    codes[0] = MOVETO;
    auto start = points;
    auto end = points + 2*point_count;
    bool closed = *start == *(end-2) && *(start+1) == *(end-1);
    if (closed) {
        std::fill(codes + 1, codes + point_count - 1, LINETO);
        codes[point_count-1] = CLOSEPOLY;
    }
    else
        std::fill(codes + 1, codes + point_count, LINETO);
}

OffsetArray Converter::convert_offsets(
    count_t offset_count, const offset_t* start, offset_t subtract)
{
    assert(offset_count > 0 && subtract >= 0);
    assert(start != nullptr);

    index_t offsets_shape = static_cast<index_t>(offset_count);
    OffsetArray py_offsets(offsets_shape);
    convert_offsets(offset_count, start, subtract, py_offsets.mutable_data());
    return py_offsets;
}

void Converter::convert_offsets(
    count_t offset_count, const offset_t* start, offset_t subtract,
    OffsetArray::value_type* offsets)
{
    assert(offset_count > 0 && subtract >= 0);
    assert(start != nullptr);
    assert(offsets != nullptr);

    check_max_offset(*(start + offset_count - 1) - subtract);

    if (subtract == 0)
        std::copy(start, start + offset_count, offsets);
    else {
        for (decltype(offset_count) i = 0; i < offset_count; ++i)
            *offsets++ = start[i] - subtract;
    }
}

PointArray Converter::convert_points(count_t point_count, const double* start)
{
    assert(point_count > 0);
    assert(start != nullptr);

    index_t points_shape[2] = {static_cast<index_t>(point_count), 2};
    PointArray py_points(points_shape);
    convert_points(point_count, start, py_points.mutable_data());
    return py_points;
}

void Converter::convert_points(count_t point_count, const double* start, double* points)
{
    assert(point_count > 0);
    assert(start != nullptr);
    assert(points != nullptr);

    std::copy(start, start + 2*point_count, points);
}

} // namespace contourpy
