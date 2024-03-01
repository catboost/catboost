#ifndef CONTOURPY_BASE_IMPL_H
#define CONTOURPY_BASE_IMPL_H

#include "base.h"
#include "converter.h"
#include "util.h"
#include <iostream>

namespace contourpy {

// Point indices from current quad index.
#define POINT_NE (quad)
#define POINT_NW (quad-1)
#define POINT_SE (quad-_nx)
#define POINT_SW (quad-_nx-1)


// CacheItem masks, only accessed directly to set.  To read, use accessors detailed below.
// 1 and 2 refer to level indices (lower and upper).
#define MASK_Z_LEVEL_1         (0x1 <<  0)  // z > lower_level.
#define MASK_Z_LEVEL_2         (0x1 <<  1)  // z > upper_level.
#define MASK_Z_LEVEL           (MASK_Z_LEVEL_1 | MASK_Z_LEVEL_2)
#define MASK_MIDDLE_Z_LEVEL_1  (0x1 <<  2)  // middle z > lower_level
#define MASK_MIDDLE_Z_LEVEL_2  (0x1 <<  3)  // middle z > upper_level
#define MASK_MIDDLE            (MASK_MIDDLE_Z_LEVEL_1 | MASK_MIDDLE_Z_LEVEL_2)
#define MASK_BOUNDARY_E        (0x1 <<  4)  // E edge of quad is a boundary.
#define MASK_BOUNDARY_N        (0x1 <<  5)  // N edge of quad is a boundary.
// EXISTS_QUAD bit is always used, but the 4 EXISTS_CORNER are only used if _corner_mask is true.
// Only one of EXISTS_QUAD or EXISTS_??_CORNER is ever set per quad.
#define MASK_EXISTS_QUAD       (0x1 <<  6)  // All of quad exists (is not masked).
#define MASK_EXISTS_NE_CORNER  (0x1 <<  7)  // NE corner exists, SW corner is masked.
#define MASK_EXISTS_NW_CORNER  (0x1 <<  8)
#define MASK_EXISTS_SE_CORNER  (0x1 <<  9)
#define MASK_EXISTS_SW_CORNER  (0x1 << 10)
#define MASK_EXISTS_ANY_CORNER (MASK_EXISTS_NE_CORNER | MASK_EXISTS_NW_CORNER | MASK_EXISTS_SE_CORNER | MASK_EXISTS_SW_CORNER)
#define MASK_EXISTS_ANY        (MASK_EXISTS_QUAD | MASK_EXISTS_ANY_CORNER)
#define MASK_START_E           (0x1 << 11)  // E to N, filled and lines.
#define MASK_START_N           (0x1 << 12)  // N to E, filled and lines.
#define MASK_START_BOUNDARY_E  (0x1 << 13)  // Lines only.
#define MASK_START_BOUNDARY_N  (0x1 << 14)  // Lines only.
#define MASK_START_BOUNDARY_S  (0x1 << 15)  // Filled and lines.
#define MASK_START_BOUNDARY_W  (0x1 << 16)  // Filled and lines.
#define MASK_START_CORNER      (0x1 << 18)  // Filled and lines.
#define MASK_START_HOLE_N      (0x1 << 17)  // N boundary of EXISTS, E to W, filled only.
#define MASK_ANY_START         (MASK_START_N | MASK_START_E | MASK_START_BOUNDARY_N | MASK_START_BOUNDARY_E | MASK_START_BOUNDARY_S | MASK_START_BOUNDARY_W | MASK_START_HOLE_N | MASK_START_CORNER)
#define MASK_LOOK_N            (0x1 << 19)
#define MASK_LOOK_S            (0x1 << 20)
#define MASK_NO_STARTS_IN_ROW  (0x1 << 21)
#define MASK_NO_MORE_STARTS    (0x1 << 22)

// Accessors for various CacheItem masks.
#define Z_LEVEL(quad)              (_cache[quad] & MASK_Z_LEVEL)
#define Z_NE                       Z_LEVEL(POINT_NE)
#define Z_NW                       Z_LEVEL(POINT_NW)
#define Z_SE                       Z_LEVEL(POINT_SE)
#define Z_SW                       Z_LEVEL(POINT_SW)
#define MIDDLE_Z_LEVEL(quad)       ((_cache[quad] & MASK_MIDDLE) >> 2)
#define BOUNDARY_E(quad)           (_cache[quad] & MASK_BOUNDARY_E)
#define BOUNDARY_N(quad)           (_cache[quad] & MASK_BOUNDARY_N)
#define BOUNDARY_S(quad)           (_cache[quad-_nx] & MASK_BOUNDARY_N)
#define BOUNDARY_W(quad)           (_cache[quad-1] & MASK_BOUNDARY_E)
#define EXISTS_QUAD(quad)          (_cache[quad] & MASK_EXISTS_QUAD)
#define EXISTS_NE_CORNER(quad)     (_cache[quad] & MASK_EXISTS_NE_CORNER)
#define EXISTS_NW_CORNER(quad)     (_cache[quad] & MASK_EXISTS_NW_CORNER)
#define EXISTS_SE_CORNER(quad)     (_cache[quad] & MASK_EXISTS_SE_CORNER)
#define EXISTS_SW_CORNER(quad)     (_cache[quad] & MASK_EXISTS_SW_CORNER)
#define EXISTS_ANY(quad)           (_cache[quad] & MASK_EXISTS_ANY)
#define EXISTS_ANY_CORNER(quad)    (_cache[quad] & MASK_EXISTS_ANY_CORNER)
#define EXISTS_E_EDGE(quad)        (_cache[quad] & (MASK_EXISTS_QUAD | MASK_EXISTS_NE_CORNER | MASK_EXISTS_SE_CORNER))
#define EXISTS_N_EDGE(quad)        (_cache[quad] & (MASK_EXISTS_QUAD | MASK_EXISTS_NW_CORNER | MASK_EXISTS_NE_CORNER))
#define EXISTS_S_EDGE(quad)        (_cache[quad] & (MASK_EXISTS_QUAD | MASK_EXISTS_SW_CORNER | MASK_EXISTS_SE_CORNER))
#define EXISTS_W_EDGE(quad)        (_cache[quad] & (MASK_EXISTS_QUAD | MASK_EXISTS_NW_CORNER | MASK_EXISTS_SW_CORNER))
// Note that EXISTS_NE_CORNER(quad) is equivalent to BOUNDARY_SW(quad), etc.
#define START_E(quad)              (_cache[quad] & MASK_START_E)
#define START_N(quad)              (_cache[quad] & MASK_START_N)
#define START_BOUNDARY_E(quad)     (_cache[quad] & MASK_START_BOUNDARY_E)
#define START_BOUNDARY_N(quad)     (_cache[quad] & MASK_START_BOUNDARY_N)
#define START_BOUNDARY_S(quad)     (_cache[quad] & MASK_START_BOUNDARY_S)
#define START_BOUNDARY_W(quad)     (_cache[quad] & MASK_START_BOUNDARY_W)
#define START_CORNER(quad)         (_cache[quad] & MASK_START_CORNER)
#define START_HOLE_N(quad)         (_cache[quad] & MASK_START_HOLE_N)
#define ANY_START(quad)            ((_cache[quad] & MASK_ANY_START) != 0)
#define LOOK_N(quad)               (_cache[quad] & MASK_LOOK_N)
#define LOOK_S(quad)               (_cache[quad] & MASK_LOOK_S)
#define NO_STARTS_IN_ROW(quad)     (_cache[quad] & MASK_NO_STARTS_IN_ROW)
#define NO_MORE_STARTS(quad)       (_cache[quad] & MASK_NO_MORE_STARTS)
// Contour line/fill goes to the left or right of quad middle (quad_as_tri only).
#define LEFT_OF_MIDDLE(quad, is_upper) (MIDDLE_Z_LEVEL(quad) == (is_upper ? 2 : 0))


template <typename Derived>
BaseContourGenerator<Derived>::BaseContourGenerator(
    const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
    const MaskArray& mask, bool corner_mask, LineType line_type, FillType fill_type,
    bool quad_as_tri, ZInterp z_interp, index_t x_chunk_size, index_t y_chunk_size)
    : _x(x),
      _y(y),
      _z(z),
      _xptr(_x.data()),
      _yptr(_y.data()),
      _zptr(_z.data()),
      _nx(_z.ndim() > 1 ? _z.shape(1) : 0),
      _ny(_z.ndim() > 0 ? _z.shape(0) : 0),
      _n(_nx*_ny),
      _x_chunk_size(x_chunk_size > 0 ? std::min(x_chunk_size, _nx-1) : _nx-1),
      _y_chunk_size(y_chunk_size > 0 ? std::min(y_chunk_size, _ny-1) : _ny-1),
      _nx_chunks(static_cast<index_t>(std::ceil((_nx-1.0) / _x_chunk_size))),
      _ny_chunks(static_cast<index_t>(std::ceil((_ny-1.0) / _y_chunk_size))),
      _n_chunks(_nx_chunks*_ny_chunks),
      _corner_mask(corner_mask),
      _line_type(line_type),
      _fill_type(fill_type),
      _quad_as_tri(quad_as_tri),
      _z_interp(z_interp),
      _cache(new CacheItem[_n]),
      _filled(false),
      _lower_level(0.0),
      _upper_level(0.0),
      _identify_holes(false),
      _output_chunked(false),
      _direct_points(false),
      _direct_line_offsets(false),
      _direct_outer_offsets(false),
      _outer_offsets_into_points(false),
      _nan_separated(false),
      _return_list_count(0)
{
    if (_x.ndim() != 2 || _y.ndim() != 2 || _z.ndim() != 2)
        throw std::invalid_argument("x, y and z must all be 2D arrays");

    if (_x.shape(1) != _nx || _x.shape(0) != _ny ||
        _y.shape(1) != _nx || _y.shape(0) != _ny)
        throw std::invalid_argument("x, y and z arrays must have the same shape");

    if (_nx < 2 || _ny < 2)
        throw std::invalid_argument("x, y and z must all be at least 2x2 arrays");

    if (mask.ndim() != 0) {  // ndim == 0 if mask is not set, which is valid.
        if (mask.ndim() != 2)
            throw std::invalid_argument("mask array must be a 2D array");

        if (mask.shape(1) != _nx || mask.shape(0) != _ny)
            throw std::invalid_argument(
                "If mask is set it must be a 2D array with the same shape as z");
    }

    if (!supports_line_type(line_type))
        throw std::invalid_argument("Unsupported LineType");

    if (!supports_fill_type(fill_type))
        throw std::invalid_argument("Unsupported FillType");

    if (x_chunk_size < 0 || y_chunk_size < 0)
        throw std::invalid_argument("x_chunk_size and y_chunk_size cannot be negative");

    if (_z_interp == ZInterp::Log) {
        const bool* mask_ptr = (mask.ndim() == 0 ? nullptr : mask.data());
        for (index_t point = 0; point < _n; ++point) {
            if ( (mask_ptr == nullptr || !mask_ptr[point]) && _zptr[point] <= 0.0)
                throw std::invalid_argument("z values must be positive if using ZInterp.Log");
        }
    }

    init_cache_grid(mask);
}

template <typename Derived>
BaseContourGenerator<Derived>::~BaseContourGenerator()
{
    delete [] _cache;
}

template <typename Derived>
typename BaseContourGenerator<Derived>::ZLevel
    BaseContourGenerator<Derived>::calc_and_set_middle_z_level(index_t quad)
{
    ZLevel zlevel = z_to_zlevel(calc_middle_z(quad));
    _cache[quad] |= (zlevel << 2);
    return zlevel;
}

template <typename Derived>
double BaseContourGenerator<Derived>::calc_middle_z(index_t quad) const
{
    assert(quad >= 0 && quad < _n);

    switch (_z_interp) {
        case ZInterp::Log:
            return exp(0.25*(log(get_point_z(POINT_SW)) +
                             log(get_point_z(POINT_SE)) +
                             log(get_point_z(POINT_NW)) +
                             log(get_point_z(POINT_NE))));
        default:  // ZInterp::Linear
            return 0.25*(get_point_z(POINT_SW) +
                         get_point_z(POINT_SE) +
                         get_point_z(POINT_NW) +
                         get_point_z(POINT_NE));
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::check_consistent_counts(const ChunkLocal& local) const
{
    if (local.total_point_count > 0) {
        if (local.points.size != 2*local.total_point_count ||
            local.points.current != local.points.start + 2*local.total_point_count) {
            throw std::runtime_error(
                "Inconsistent total_point_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }
    else {
        if (local.points.size != 0 ||
            local.points.start != nullptr || local.points.current != nullptr) {
            throw std::runtime_error(
                "Inconsistent zero total_point_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }

    if (local.line_count > 0) {
        if (local.line_offsets.size != local.line_count + 1 ||
            local.line_offsets.current == nullptr ||
            local.line_offsets.current != local.line_offsets.start + local.line_count + 1) {
            throw std::runtime_error(
                "Inconsistent line_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }
    else {
        if (local.line_offsets.size != 0 ||
            local.line_offsets.start != nullptr || local.line_offsets.current != nullptr) {
            throw std::runtime_error(
                "Inconsistent zero line_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }

    if (_identify_holes && local.line_count > 0) {
        if (local.outer_offsets.size != local.line_count - local.hole_count + 1 ||
            local.outer_offsets.current == nullptr ||
            local.outer_offsets.current != local.outer_offsets.start + local.line_count -
                                           local.hole_count + 1) {
            throw std::runtime_error(
                "Inconsistent hole_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }
    else {
        if (local.outer_offsets.size != 0 ||
            local.outer_offsets.start != nullptr || local.outer_offsets.current != nullptr) {
            throw std::runtime_error(
                "Inconsistent zero hole_count for chunk " + std::to_string(local.chunk) +
                ". This may indicate a bug in ContourPy.");
        }
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::closed_line(
    const Location& start_location, OuterOrHole outer_or_hole, ChunkLocal& local)
{
    assert(is_quad_in_chunk(start_location.quad, local));

    Location location = start_location;
    bool finished = false;
    count_t point_count = 0;

    if (outer_or_hole == Hole && local.pass == 0 && _identify_holes)
        set_look_flags(start_location.quad);

    while (!finished) {
        if (location.on_boundary)
            finished = follow_boundary(location, start_location, local, point_count);
        else
            finished = follow_interior(location, start_location, local, point_count);
        location.on_boundary = !location.on_boundary;
    }

    if (local.pass > 0) {
        assert(local.line_offsets.current = local.line_offsets.start + local.line_count);
        *local.line_offsets.current++ = local.total_point_count;
        if (outer_or_hole == Outer && _identify_holes) {
            assert(local.outer_offsets.current ==
                local.outer_offsets.start + local.line_count - local.hole_count);
            if (_outer_offsets_into_points)
                *local.outer_offsets.current++ = local.total_point_count;
            else
                *local.outer_offsets.current++ = local.line_count;
        }
    }

    local.total_point_count += point_count;
    local.line_count++;
    if (outer_or_hole == Hole)
        local.hole_count++;
}

template <typename Derived>
void BaseContourGenerator<Derived>::closed_line_wrapper(
    const Location& start_location, OuterOrHole outer_or_hole, ChunkLocal& local)
{
    assert(is_quad_in_chunk(start_location.quad, local));

    if (local.pass == 0 || !_identify_holes) {
        closed_line(start_location, outer_or_hole, local);
    }
    else {
        assert(outer_or_hole == Outer);
        local.look_up_quads.clear();

        closed_line(start_location, outer_or_hole, local);

        for (py::size_t i = 0; i < local.look_up_quads.size(); ++i) {
            // Note that the collection can increase in size during this loop.
            index_t quad = local.look_up_quads[i];

            // Walk N to corresponding look S flag is reached.
            quad = find_look_S(quad);

            // Only 3 possible types of hole start: START_E, START_HOLE_N or START_CORNER for SW
            // corner.
            if (START_E(quad)) {
                closed_line(Location(quad, -1, -_nx, Z_NE > 0, false), Hole, local);
            }
            else if (START_HOLE_N(quad)) {
                closed_line(Location(quad, -1, -_nx, false, true), Hole, local);
            }
            else {
                assert(START_CORNER(quad) && EXISTS_SW_CORNER(quad));
                closed_line(Location(quad, _nx-1, -_nx-1, false, true), Hole, local);
            }
        }
    }
}

template <typename Derived>
FillType BaseContourGenerator<Derived>::default_fill_type()
{
    FillType fill_type = FillType::OuterOffset;
    assert(supports_fill_type(fill_type));
    return fill_type;
}

template <typename Derived>
LineType BaseContourGenerator<Derived>::default_line_type()
{
    LineType line_type = LineType::Separate;
    assert(supports_line_type(line_type));
    return line_type;
}

template <typename Derived>
py::sequence BaseContourGenerator<Derived>::filled(double lower_level, double upper_level)
{
    if (lower_level >= upper_level)
        throw std::invalid_argument("upper_level must be larger than lower_level");

    _filled = true;
    _lower_level = lower_level;
    _upper_level = upper_level;

    _identify_holes = !(_fill_type == FillType::ChunkCombinedCode ||
                        _fill_type == FillType::ChunkCombinedOffset);
    _output_chunked = !(_fill_type == FillType::OuterCode || _fill_type == FillType::OuterOffset);
    _direct_points = _output_chunked;
    _direct_line_offsets = (_fill_type == FillType::ChunkCombinedOffset||
                            _fill_type == FillType::ChunkCombinedOffsetOffset);
    _direct_outer_offsets = (_fill_type == FillType::ChunkCombinedCodeOffset ||
                             _fill_type == FillType::ChunkCombinedOffsetOffset);
    _outer_offsets_into_points = (_fill_type == FillType::ChunkCombinedCodeOffset);
    _nan_separated = false;
    _return_list_count = (_fill_type == FillType::ChunkCombinedCodeOffset ||
                          _fill_type == FillType::ChunkCombinedOffsetOffset) ? 3 : 2;

    return march_wrapper();
}

template <typename Derived>
index_t BaseContourGenerator<Derived>::find_look_S(index_t look_N_quad) const
{
    assert(_identify_holes);

    // Might need to be careful when looking in the same quad as the LOOK_UP.
    index_t quad = look_N_quad;

    // look_S quad must have 1 of only 3 possible types of hole start (START_E, START_HOLE_N,
    // START_CORNER for SW corner) but it may have other starts as well.

    // Start quad may be both a look_N and look_S quad.  Only want to stop search here if look_S
    // hole start is N of look_N.

    if (!LOOK_S(quad)) {
        do
        {
            quad += _nx;
            assert(quad >= 0 && quad < _n);
            assert(EXISTS_ANY(quad));
        } while (!LOOK_S(quad));
    }

    return quad;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::follow_boundary(
    Location& location, const Location& start_location, ChunkLocal& local, count_t& point_count)
{
    // forward values for boundaries:
    //     -1 = N boundary, E to W.
    //      1 = S boundary, W to E.
    //   -_nx = W boundary, N to S.
    //    _nx = E boundary, S to N.
    // -_nx+1 = NE corner, NW to SE.
    //  _nx+1 = NW corner, SW to NE.
    // -_nx-1 = SE corner, NE to SW.
    //  _nx-1 = SW corner, SE to NW.

    assert(is_quad_in_chunk(start_location.quad, local));
    assert(is_quad_in_chunk(location.quad, local));

    // Local variables for faster access.
    auto quad = location.quad;
    auto forward = location.forward;
    auto left = location.left;
    auto start_quad = start_location.quad;
    auto start_forward = start_location.forward;
    auto start_left = start_location.left;
    auto pass = local.pass;
    double*& points = local.points.current;

    auto start_point = get_boundary_start_point(location);
    auto end_point = start_point + forward;

    assert(is_point_in_chunk(start_point, local));
    assert(is_point_in_chunk(end_point, local));

    auto start_z = Z_LEVEL(start_point);
    auto end_z = Z_LEVEL(end_point);

    // Add new point, somewhere along start line.  May be at start point of edge if this is a
    // boundary start.
    point_count++;
    if (pass > 0) {
        if (start_z == 1)
            get_point_xy(start_point, points);
        else  // start_z != 1
            interp(start_point, end_point, location.is_upper, points);
    }

    bool finished = false;
    while (true) {
        assert(is_quad_in_chunk(quad, local));

        if (quad == start_quad && forward == start_forward && left == start_left) {
            if (start_location.on_boundary && point_count > 1) {
                // Polygon closed.
                finished = true;
                break;
            }
        }
        else if (pass == 0) {
            // Clear unwanted start locations.
            if (left == _nx) {
                if (START_BOUNDARY_S(quad)) {
                    assert(forward == 1);
                    _cache[quad] &= ~MASK_START_BOUNDARY_S;
                }
            }
            else if (forward == -_nx) {
                if (START_BOUNDARY_W(quad)) {
                    assert(left == 1);
                    _cache[quad] &= ~MASK_START_BOUNDARY_W;
                }
            }
            else if (left == -_nx) {
                if (START_HOLE_N(quad)) {
                    assert(forward == -1);
                    _cache[quad] &= ~MASK_START_HOLE_N;
                }
            }
            else {
                switch (EXISTS_ANY_CORNER(quad)) {
                    case MASK_EXISTS_NE_CORNER:
                        if (left == _nx+1) {
                            assert(forward == -_nx+1);
                            _cache[quad] &= ~MASK_START_CORNER;
                        }
                        break;
                    case MASK_EXISTS_NW_CORNER:
                        if (forward == _nx+1) {
                            assert(left == _nx-1);
                            _cache[quad] &= ~MASK_START_CORNER;
                        }
                        break;
                    case MASK_EXISTS_SE_CORNER:
                        if (forward == -_nx-1) {
                            assert(left == -_nx+1);
                            _cache[quad] &= ~MASK_START_CORNER;
                        }
                        break;
                    case MASK_EXISTS_SW_CORNER:
                        if (left == -_nx-1) {
                            assert(forward == _nx-1);
                            _cache[quad] &= ~MASK_START_CORNER;
                        }
                        break;
                    default:
                        // Not a corner.
                        break;
                }
            }
        }

        // Check if need to leave boundary into interior.
        if (end_z != 1) {
            location.is_upper = (end_z == 2);  // Leave via this level.
            auto temp = forward;
            forward = left;
            left = -temp;
            break;
        }

        // Add end point.
        point_count++;
        if (pass > 0) {
            get_point_xy(end_point, points);

            if (LOOK_N(quad) && _identify_holes &&
                (left == _nx || left == _nx+1 || forward == _nx+1)) {
                assert(BOUNDARY_N(quad-_nx) || EXISTS_NE_CORNER(quad) || EXISTS_NW_CORNER(quad));
                local.look_up_quads.push_back(quad);
            }
        }

        move_to_next_boundary_edge(quad, forward, left);

        start_point = end_point;
        start_z = end_z;
        end_point = start_point + forward;
        end_z = Z_LEVEL(end_point);
    }

    location.quad = quad;
    location.forward = forward;
    location.left = left;

    return finished;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::follow_interior(
    Location& location, const Location& start_location, ChunkLocal& local, count_t& point_count)
{
    // Adds the start point in each quad visited, but not the end point unless closing the polygon.
    // Only need to consider a single level of course.
    assert(is_quad_in_chunk(start_location.quad, local));
    assert(is_quad_in_chunk(location.quad, local));

    // Local variables for faster access.
    auto quad = location.quad;
    auto forward = location.forward;
    auto left = location.left;
    auto is_upper = location.is_upper;
    auto start_quad = start_location.quad;
    auto start_forward = start_location.forward;
    auto start_left = start_location.left;
    auto pass = local.pass;
    double*& points = local.points.current;

    // left direction, and indices of points on entry edge.
    bool start_corner_diagonal = false;
    auto left_point = get_interior_start_left_point(location, start_corner_diagonal);
    auto right_point = left_point - left;
    bool want_look_N = _identify_holes && pass > 0;

    bool finished = false;  // Whether finished line, i.e. returned to start.
    while (true) {
        assert(is_quad_in_chunk(quad, local));
        assert(is_point_in_chunk(left_point, local));
        assert(is_point_in_chunk(right_point, local));

        if (pass > 0)
            interp(left_point, right_point, is_upper, points);
        point_count++;

        if (quad == start_quad && forward == start_forward &&
            left == start_left && is_upper == start_location.is_upper &&
            !start_location.on_boundary && point_count > 1) {
            finished = true;  // Polygon closed, exit immediately.
            break;
        }

        // Indices of the opposite points.
        auto opposite_left_point = left_point + forward;
        auto opposite_right_point = right_point + forward;
        bool corner_opposite_is_right = false;  // Only used for corners.

        if (start_corner_diagonal) {
            // To avoid dealing with diagonal forward and left below, switch to direction 45 degrees
            // to left, e.g. NW corner faces west using forward == -1.
            corner_opposite_is_right = true;
            switch (EXISTS_ANY_CORNER(quad)) {
                case MASK_EXISTS_NW_CORNER:
                    forward = -1;
                    left = -_nx;
                    opposite_left_point = opposite_right_point = quad-1;
                    break;
                case MASK_EXISTS_NE_CORNER:
                    forward = _nx;
                    left = -1;
                    opposite_left_point = opposite_right_point = quad;
                    break;
                case MASK_EXISTS_SW_CORNER:
                    forward = -_nx;
                    left = 1;
                    opposite_left_point = opposite_right_point = quad-_nx-1;
                    break;
                default:
                    assert(EXISTS_SE_CORNER(quad));
                    forward = 1;
                    left = _nx;
                    opposite_left_point = opposite_right_point = quad-_nx;
                    break;
            }
        }

        // z-levels of the opposite points.
        ZLevel z_opposite_left = Z_LEVEL(opposite_left_point);
        ZLevel z_opposite_right = Z_LEVEL(opposite_right_point);

        Direction direction = Direction::Right;
        ZLevel z_test = is_upper ? 2 : 0;

        if (EXISTS_QUAD(quad)) {
            if (z_opposite_left == z_test) {
                if (z_opposite_right == z_test || MIDDLE_Z_LEVEL(quad) == z_test)
                    direction = Direction::Left;
            }
            else if (z_opposite_right == z_test)
                direction = Direction::Straight;
        }
        else if (start_corner_diagonal) {
            direction = (z_opposite_left == z_test) ? Direction::Straight : Direction::Right;
        }
        else {
            switch (EXISTS_ANY_CORNER(quad)) {
                case MASK_EXISTS_NW_CORNER:
                    corner_opposite_is_right = (forward == -_nx);
                    break;
                case MASK_EXISTS_NE_CORNER:
                    corner_opposite_is_right = (forward == -1);
                    break;
                case MASK_EXISTS_SW_CORNER:
                    corner_opposite_is_right = (forward == 1);
                    break;
                default:
                    assert(EXISTS_SE_CORNER(quad));
                    corner_opposite_is_right = (forward == _nx);
                    break;
            }

            if (corner_opposite_is_right)
                direction = (z_opposite_right == z_test) ? Direction::Straight : Direction::Right;
            else
                direction = (z_opposite_left == z_test) ? Direction::Left : Direction::Straight;
        }

        // Clear unwanted start locations.
        if (pass == 0 && !(quad == start_quad && forward == start_forward && left == start_left)) {
            if (START_E(quad) && forward == -1 && left == -_nx && direction == Direction::Right &&
                (is_upper ? Z_NE > 0 : Z_NE < 2)) {
                _cache[quad] &= ~MASK_START_E;  // E high if is_upper else low.

                if (!_filled && quad < start_location.quad)
                    // Already counted points from here onwards.
                    break;
            }
            else if (START_N(quad) && forward == -_nx && left == 1 &&
                     direction == Direction::Left && (is_upper ? Z_NW > 0 : Z_NW < 2)) {
                _cache[quad] &= ~MASK_START_N;  // E high if is_upper else low.

                if (!_filled && quad < start_location.quad)
                    // Already counted points from here onwards.
                    break;
            }
        }

        // Extra quad_as_tri points.
        if (_quad_as_tri && EXISTS_QUAD(quad)) {
            if (pass == 0) {
                switch (direction) {
                    case Direction::Left:
                        point_count += (LEFT_OF_MIDDLE(quad, is_upper) ? 1 : 3);
                        break;
                    case Direction::Right:
                        point_count += (LEFT_OF_MIDDLE(quad, is_upper) ? 3 : 1);
                        break;
                    case Direction::Straight:
                        point_count += 2;
                        break;
                }
            }
            else {  // pass == 1
                auto mid_x = get_middle_x(quad);
                auto mid_y = get_middle_y(quad);
                auto mid_z = calc_middle_z(quad);

                switch (direction) {
                    case Direction::Left:
                        if (LEFT_OF_MIDDLE(quad, is_upper)) {
                            interp(left_point, mid_x, mid_y, mid_z, is_upper, points);
                            point_count++;
                        }
                        else {
                            interp(right_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_right_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_left_point, mid_x, mid_y, mid_z, is_upper, points);
                            point_count += 3;
                        }
                        break;
                    case Direction::Right:
                        if (LEFT_OF_MIDDLE(quad, is_upper)) {
                            interp(left_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_left_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_right_point, mid_x, mid_y, mid_z, is_upper, points);
                            point_count += 3;
                        }
                        else {
                            interp(right_point, mid_x, mid_y, mid_z, is_upper, points);
                            point_count++;
                        }
                        break;
                    case Direction::Straight:
                        if (LEFT_OF_MIDDLE(quad, is_upper)) {
                            interp(left_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_left_point, mid_x, mid_y, mid_z, is_upper, points);
                        }
                        else {
                            interp(right_point, mid_x, mid_y, mid_z, is_upper, points);
                            interp(opposite_right_point, mid_x, mid_y, mid_z, is_upper, points);
                        }
                        point_count += 2;
                        break;
                }
            }
        }

        bool reached_boundary = false;

        // Determine entry edge and left and right points of next quad.
        // Do not update quad index yet.
        switch (direction) {
            case Direction::Left: {
                auto temp = forward;
                forward = left;
                left = -temp;
                // left_point unchanged.
                right_point = opposite_left_point;
                break;
            }
            case Direction::Right: {
                auto temp = forward;
                forward = -left;
                left = temp;
                left_point = opposite_right_point;
                // right_point unchanged.
                break;
            }
            case Direction::Straight: {
                if (EXISTS_QUAD(quad)) {  // Straight on in quad.
                    // forward and left stay the same.
                    left_point = opposite_left_point;
                    right_point = opposite_right_point;
                }
                else if (start_corner_diagonal) {  // Straight on diagonal start corner.
                    // left point unchanged.
                    right_point = opposite_right_point;
                }
                else {  // Straight on in a corner reaches boundary.
                    assert(EXISTS_ANY_CORNER(quad));
                    reached_boundary = true;

                    if (corner_opposite_is_right) {
                        // left_point unchanged.
                        right_point = opposite_right_point;
                    }
                    else {
                        left_point = opposite_left_point;
                        // right_point unchanged.
                    }

                    // Set forward and left for correct exit along boundary.
                    switch (EXISTS_ANY_CORNER(quad)) {
                        case MASK_EXISTS_NW_CORNER:
                            forward = _nx+1;
                            left = _nx-1;
                            break;
                        case MASK_EXISTS_NE_CORNER:
                            forward = -_nx+1;
                            left = _nx+1;
                            break;
                        case MASK_EXISTS_SW_CORNER:
                            forward = _nx-1;
                            left = -_nx-1;
                            break;
                        default:
                            assert(EXISTS_SE_CORNER(quad));
                            forward = -_nx-1;
                            left = -_nx+1;
                            break;
                   }
                }
                break;
            }
        }

        if (want_look_N && LOOK_N(quad) && forward == 1) {
            // Only consider look_N if pass across E edge of this quad.
            // Care needed if both look_N and look_S set in quad because this line corresponds to
            // only one of them, so want to ignore the look_N if it is the other line otherwise it
            // will be double counted.
            if (!LOOK_S(quad) || (is_upper ? Z_NE < 2 : Z_NE > 0))
                local.look_up_quads.push_back(quad);
        }

        // Check if reached NSEW boundary; already checked and noted if reached corner boundary.
        if (!reached_boundary) {
            if (forward > 0)
                reached_boundary = (forward == 1 ? BOUNDARY_E(quad) : BOUNDARY_N(quad));
            else  // forward < 0
                reached_boundary = (forward == -1 ? BOUNDARY_W(quad) : BOUNDARY_S(quad));

            if (reached_boundary) {
                auto temp = forward;
                forward = left;
                left = -temp;
            }
        }

        // If reached a boundary, return.
        if (reached_boundary) {
            if (!_filled) {
                point_count++;
                if (pass > 0)
                    interp(left_point, right_point, false, points);
            }
            break;
        }

        quad += forward;
        start_corner_diagonal = false;
    }

    location.quad = quad;
    location.forward = forward;
    location.left = left;
    location.is_upper = is_upper;

    return finished;
}

template <typename Derived>
index_t BaseContourGenerator<Derived>::get_boundary_start_point(const Location& location) const
{
    auto quad = location.quad;
    auto forward = location.forward;
    auto left = location.left;
    index_t start_point = -1;

    if (forward > 0) {
        if (forward == _nx) {
            assert(left == -1);
            start_point = quad-_nx;
        }
        else if (left == _nx) {
            assert(forward == 1);
            start_point = quad-_nx-1;
        }
        else if (EXISTS_SW_CORNER(quad)) {
            assert(forward == _nx-1 && left == -_nx-1);
            start_point = quad-_nx;
        }
        else {
            assert(EXISTS_NW_CORNER(quad) && forward == _nx+1 && left == _nx-1);
            start_point = quad-_nx-1;
        }
    }
    else {  // forward < 0
        if (forward == -_nx) {
            assert(left == 1);
            start_point = quad-1;
        }
        else if (left == -_nx) {
            assert(forward == -1);
            start_point = quad;
        }
        else if (EXISTS_NE_CORNER(quad)) {
            assert(forward == -_nx+1 && left == _nx+1);
            start_point = quad-1;
        }
        else {
            assert(EXISTS_SE_CORNER(quad) && forward == -_nx-1 && left == -_nx+1);
            start_point = quad;
        }
    }
    return start_point;
}

template <typename Derived>
py::tuple BaseContourGenerator<Derived>::get_chunk_count() const
{
    return py::make_tuple(_ny_chunks, _nx_chunks);
}

template <typename Derived>
void BaseContourGenerator<Derived>::get_chunk_limits(index_t chunk, ChunkLocal& local) const
{
    assert(chunk >= 0 && chunk < _n_chunks && "chunk index out of bounds");

    local.chunk = chunk;

    auto ichunk = chunk % _nx_chunks;
    auto jchunk = chunk / _nx_chunks;

    local.istart = ichunk*_x_chunk_size + 1;
    local.iend = (ichunk < _nx_chunks-1 ? (ichunk+1)*_x_chunk_size : _nx-1);

    local.jstart = jchunk*_y_chunk_size + 1;
    local.jend = (jchunk < _ny_chunks-1 ? (jchunk+1)*_y_chunk_size : _ny-1);
}

template <typename Derived>
py::tuple BaseContourGenerator<Derived>::get_chunk_size() const
{
    return py::make_tuple(_y_chunk_size, _x_chunk_size);
}

template <typename Derived>
bool BaseContourGenerator<Derived>::get_corner_mask() const
{
    return _corner_mask;
}

template <typename Derived>
FillType BaseContourGenerator<Derived>::get_fill_type() const
{
    return _fill_type;
}

template <typename Derived>
index_t BaseContourGenerator<Derived>::get_interior_start_left_point(
    const Location& location, bool& start_corner_diagonal) const
{
    auto quad = location.quad;
    auto forward = location.forward;
    auto left = location.left;
    index_t left_point = -1;

    if (forward > 0) {
        if (forward == _nx) {
            assert(left == -1);
            left_point = quad-_nx-1;
        }
        else if (left == _nx) {
            assert(forward == 1);
            left_point = quad-1;
        }
        else if (EXISTS_NW_CORNER(quad)) {
            assert(forward == _nx-1 && left == -_nx-1);
            left_point = quad-_nx-1;
            start_corner_diagonal = true;
        }
        else {
            assert(EXISTS_NE_CORNER(quad) && forward == _nx+1 && left == _nx-1);
            left_point = quad-1;
            start_corner_diagonal = true;
        }
    }
    else {  // forward < 0
        if (forward == -_nx) {
            assert(left == 1);
            left_point = quad;
        }
        else if (left == -_nx) {
            assert(forward == -1);
            left_point = quad-_nx;
        }
        else if (EXISTS_SW_CORNER(quad)) {
            assert(forward == -_nx-1 && left == -_nx+1);
            left_point = quad-_nx;
            start_corner_diagonal = true;
        }
        else {
            assert(EXISTS_SE_CORNER(quad) && forward == -_nx+1 && left == _nx+1);
            left_point = quad;
            start_corner_diagonal = true;
        }
    }
    return left_point;
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_interp_fraction(double z0, double z1, double level) const
{
    switch (_z_interp) {
        case ZInterp::Log:
            // Equivalent to
            //   (log(z1) - log(level)) / (log(z1) - log(z0))
            // Same result obtained regardless of logarithm base.
            return log(z1/level) / log(z1/z0);
        default:  // ZInterp::Linear
            return (z1 - level) / (z1 - z0);
    }
}

template <typename Derived>
LineType BaseContourGenerator<Derived>::get_line_type() const
{
    return _line_type;
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_middle_x(index_t quad) const
{
    return 0.25*(get_point_x(POINT_SW) + get_point_x(POINT_SE) +
                 get_point_x(POINT_NW) + get_point_x(POINT_NE));
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_middle_y(index_t quad) const
{
    return 0.25*(get_point_y(POINT_SW) + get_point_y(POINT_SE) +
                 get_point_y(POINT_NW) + get_point_y(POINT_NE));
}

template <typename Derived>
index_t BaseContourGenerator<Derived>::get_n_chunks() const
{
    return _n_chunks;
}

template <typename Derived>
void BaseContourGenerator<Derived>::get_point_xy(index_t point, double*& points) const
{
    assert(point >= 0 && point < _n && "point index out of bounds");
    *points++ = _xptr[point];
    *points++ = _yptr[point];
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_point_x(index_t point) const
{
    assert(point >= 0 && point < _n && "point index out of bounds");
    return _xptr[point];
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_point_y(index_t point) const
{
    assert(point >= 0 && point < _n && "point index out of bounds");
    return _yptr[point];
}

template <typename Derived>
double BaseContourGenerator<Derived>::get_point_z(index_t point) const
{
    assert(point >= 0 && point < _n && "point index out of bounds");
    return _zptr[point];
}

template <typename Derived>
bool BaseContourGenerator<Derived>::get_quad_as_tri() const
{
    return _quad_as_tri;
}

template <typename Derived>
ZInterp BaseContourGenerator<Derived>::get_z_interp() const
{
    return _z_interp;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::has_direct_line_offsets() const
{
    return _direct_line_offsets;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::has_direct_outer_offsets() const
{
    return _direct_outer_offsets;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::has_direct_points() const
{
    return _direct_points;
}

template <typename Derived>
void BaseContourGenerator<Derived>::init_cache_grid(const MaskArray& mask)
{
    index_t i, j, quad;
    if (mask.ndim() == 0) {
        // No mask, easy to calculate quad existence and boundaries together.
        for (j = 0, quad = 0; j < _ny; ++j) {
            for (i = 0; i < _nx; ++i, ++quad) {
                _cache[quad] = 0;

                if (i > 0 && j > 0)
                    _cache[quad] |= MASK_EXISTS_QUAD;

                if ((i % _x_chunk_size == 0 || i == _nx-1) && j > 0)
                    _cache[quad] |= MASK_BOUNDARY_E;

                if ((j % _y_chunk_size == 0 || j == _ny-1) && i > 0)
                    _cache[quad] |= MASK_BOUNDARY_N;
            }
        }
    }
    else {
        // Could maybe speed this up and just have a single pass.
        // Care would be needed with lookback of course.
        const bool* mask_ptr = mask.data();

        // Have mask so use two stages.
        // Stage 1, determine if quads/corners exist.
        quad = 0;
        for (j = 0; j < _ny; ++j) {
            for (i = 0; i < _nx; ++i, ++quad) {
                _cache[quad] = 0;

                if (i > 0 && j > 0) {
                    unsigned int config = (mask_ptr[POINT_NW] << 3) |
                                          (mask_ptr[POINT_NE] << 2) |
                                          (mask_ptr[POINT_SW] << 1) |
                                          (mask_ptr[POINT_SE] << 0);
                    if (_corner_mask) {
                         switch (config) {
                            case 0: _cache[quad] = MASK_EXISTS_QUAD; break;
                            case 1: _cache[quad] = MASK_EXISTS_NW_CORNER; break;
                            case 2: _cache[quad] = MASK_EXISTS_NE_CORNER; break;
                            case 4: _cache[quad] = MASK_EXISTS_SW_CORNER; break;
                            case 8: _cache[quad] = MASK_EXISTS_SE_CORNER; break;
                            default:
                                // Do nothing, quad is masked out.
                                break;
                        }
                    }
                    else if (config == 0)
                        _cache[quad] = MASK_EXISTS_QUAD;
                }
            }
        }

        // Stage 2, calculate N and E boundaries.
        quad = 0;
        for (j = 0; j < _ny; ++j) {
            bool j_chunk_boundary = j % _y_chunk_size == 0;

            for (i = 0; i < _nx; ++i, ++quad) {
                bool i_chunk_boundary = i % _x_chunk_size == 0;

                if (_corner_mask) {
                    bool exists_E_edge = EXISTS_E_EDGE(quad);
                    bool E_exists_W_edge = (i < _nx-1 && EXISTS_W_EDGE(quad+1));
                    bool exists_N_edge = EXISTS_N_EDGE(quad);
                    bool N_exists_S_edge = (j < _ny-1 && EXISTS_S_EDGE(quad+_nx));

                    if (exists_E_edge != E_exists_W_edge ||
                        (i_chunk_boundary && exists_E_edge && E_exists_W_edge))
                        _cache[quad] |= MASK_BOUNDARY_E;

                    if (exists_N_edge != N_exists_S_edge ||
                        (j_chunk_boundary && exists_N_edge && N_exists_S_edge))
                         _cache[quad] |= MASK_BOUNDARY_N;
                }
                else {
                    bool E_exists_quad = (i < _nx-1 && EXISTS_QUAD(quad+1));
                    bool N_exists_quad = (j < _ny-1 && EXISTS_QUAD(quad+_nx));
                    bool exists = EXISTS_QUAD(quad);

                    if (exists != E_exists_quad || (i_chunk_boundary && exists && E_exists_quad))
                        _cache[quad] |= MASK_BOUNDARY_E;

                    if (exists != N_exists_quad || (j_chunk_boundary && exists && N_exists_quad))
                        _cache[quad] |= MASK_BOUNDARY_N;
                }
            }
        }
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::init_cache_levels_and_starts(const ChunkLocal* local)
{
    bool ordered_chunks = (local == nullptr);

    // This function initialises the cache z-levels and starts for either a single chunk or the
    // whole domain.  If a single chunk, only the quads contained in the chunk are calculated and
    // this includes the z-levels of the points that on the NE corners of those quads.  In addition,
    // chunks that are on the W (starting at i=1) also calculate the most westerly points (i=0),
    // and similarly chunks that are on the S (starting at j=1) also calculate the most southerly
    // points (j=0).  Non W/S chunks do not do this as their neighboring chunks to the W/S are
    // responsible for it.  If ordered_chunks is true then those W/S points will already have had
    // their cache items set so that their z-levels can be read from the cache as usual.  But if
    // ordered_chunks is false then we cannot rely upon those neighboring W/S points having their
    // cache items already set and so must temporarily calculate those z-levels rather than reading
    // the cache.

    constexpr CacheItem keep_mask = (MASK_EXISTS_ANY | MASK_BOUNDARY_N | MASK_BOUNDARY_E);

    index_t istart, iend, jstart, jend;  // Loop indices.
    index_t chunk_istart;  // Actual start i-index of chunk.

    if (local != nullptr) {
        chunk_istart = local->istart;
        istart = chunk_istart > 1 ? chunk_istart : 0;
        iend = local->iend;
        jstart = local->jstart > 1 ? local->jstart : 0;
        jend = local->jend;
    }
    else {
        chunk_istart = 1;
        istart = 0;
        iend = _nx-1;
        jstart = 0;
        jend = _ny-1;
    }

    index_t j_final_start = jstart - 1;
    bool calc_W_z_level = (!ordered_chunks && istart == chunk_istart);

    for (index_t j = jstart; j <= jend; ++j) {
        index_t quad = istart + j*_nx;
        const double* z_ptr = _zptr + quad;
        bool start_in_row = false;
        bool calc_S_z_level = (!ordered_chunks && j == jstart);

        // z-level of NW point not needed if i == 0.
        ZLevel z_nw = (istart == 0) ? 0 : (calc_W_z_level ? z_to_zlevel(*(z_ptr-1)) : Z_NW);

        // z-level of SW point not needed if i == 0 or j == 0.
        ZLevel z_sw = (istart == 0 || j == 0) ? 0 :
            ((calc_W_z_level || calc_S_z_level) ? z_to_zlevel(*(z_ptr-_nx-1)) : Z_SW);

        for (index_t i = istart; i <= iend; ++i, ++quad, ++z_ptr) {
            // z-level of SE point not needed if j == 0.
            ZLevel z_se = (j == 0) ? 0 : (calc_S_z_level ? z_to_zlevel(*(z_ptr-_nx)) : Z_SE);

            _cache[quad] &= keep_mask;

            // Calculate and cache z-level of NE point.
            ZLevel z_ne = z_to_zlevel(*z_ptr);
            _cache[quad] |= z_ne;

            switch (EXISTS_ANY(quad)) {
                case MASK_EXISTS_QUAD:
                    if (_filled) {
                        switch ((z_nw << 6) | (z_ne << 4) | (z_sw << 2) | z_se) {  // config
                            case   1:  // 0001
                            case   2:  // 0002
                            case  17:  // 0101
                            case  18:  // 0102
                            case  34:  // 0202
                            case  68:  // 1010
                            case 102:  // 1212
                            case 136:  // 2020
                            case 152:  // 2120
                            case 153:  // 2121
                            case 168:  // 2220
                            case 169:  // 2221
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case   4:  // 0010
                            case   5:  // 0011
                            case   6:  // 0012
                            case   8:  // 0020
                            case   9:  // 0021
                            case  21:  // 0111
                            case  22:  // 0112
                            case  25:  // 0121
                            case  38:  // 0212
                            case  72:  // 1020
                            case  98:  // 1202
                            case 132:  // 2010
                            case 145:  // 2101
                            case 148:  // 2110
                            case 149:  // 2111
                            case 161:  // 2201
                            case 162:  // 2202
                            case 164:  // 2210
                            case 165:  // 2211
                            case 166:  // 2212
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  10:  // 0022
                            case  26:  // 0122
                            case  42:  // 0222
                            case  64:  // 1000
                            case 106:  // 1222
                            case 128:  // 2000
                            case 144:  // 2100
                            case 160:  // 2200
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                            case  16:  // 0100
                            case 154:  // 2122
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                _cache[quad] |= MASK_START_N;
                                start_in_row = true;
                                break;
                            case  20:  // 0110
                            case  24:  // 0120
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) == 0) _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  32:  // 0200
                            case 138:  // 2022
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                _cache[quad] |= MASK_START_E;
                                _cache[quad] |= MASK_START_N;
                                start_in_row = true;
                                break;
                            case  33:  // 0201
                            case  69:  // 1011
                            case  70:  // 1012
                            case 100:  // 1210
                            case 101:  // 1211
                            case 137:  // 2021
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case  36:  // 0210
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) == 0) _cache[quad] |= MASK_START_N;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case  37:  // 0211
                            case  73:  // 1021
                            case  97:  // 1201
                            case 133:  // 2011
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case  40:  // 0220
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) < 2) _cache[quad] |= MASK_START_E;
                                if (MIDDLE_Z_LEVEL(quad) == 0) _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  41:  // 0221
                            case 104:  // 1220
                            case 105:  // 1221
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) < 2) _cache[quad] |= MASK_START_E;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  65:  // 1001
                            case  66:  // 1002
                            case 129:  // 2001
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) > 0) _cache[quad] |= MASK_START_E;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  74:  // 1022
                            case  96:  // 1200
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case  80:  // 1100
                            case  90:  // 1122
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  81:  // 1101
                            case  82:  // 1102
                            case  88:  // 1120
                            case  89:  // 1121
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  84:  // 1110
                            case  85:  // 1111
                            case  86:  // 1112
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case 130:  // 2002
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) > 0) _cache[quad] |= MASK_START_E;
                                if (MIDDLE_Z_LEVEL(quad) == 2) _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case 134:  // 2012
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) == 2) _cache[quad] |= MASK_START_N;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case 146:  // 2102
                            case 150:  // 2112
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (MIDDLE_Z_LEVEL(quad) == 2) _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                        }
                    }
                    else {  // !_filled quad
                        switch ((z_nw << 3) | (z_ne << 2) | (z_sw << 1) | z_se) {  // config
                            case  1:  // 0001
                            case  3:  // 0011
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_E(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                    start_in_row = true;
                                }
                                break;
                            case  2:  // 0010
                            case 10:  // 1010
                            case 14:  // 1110
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case  4:  // 0100
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_N(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                else if (!BOUNDARY_E(quad))
                                    _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  5:  // 0101
                            case  7:  // 0111
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_N(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                    start_in_row = true;
                                }
                                break;
                            case  6:  // 0110
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_N(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                else if (!BOUNDARY_E(quad) && MIDDLE_Z_LEVEL(quad) == 0)
                                    _cache[quad] |= MASK_START_N;
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                start_in_row |= ANY_START(quad);
                                break;
                            case  8:  // 1000
                            case 12:  // 1100
                            case 13:  // 1101
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                            case  9:  // 1001
                                calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_E(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                else if (!BOUNDARY_N(quad) && MIDDLE_Z_LEVEL(quad) == 1)
                                    _cache[quad] |= MASK_START_E;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                start_in_row |= ANY_START(quad);
                                break;
                            case 11:  // 1011
                                if (_quad_as_tri) calc_and_set_middle_z_level(quad);
                                if (BOUNDARY_E(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                else if (!BOUNDARY_N(quad))
                                    _cache[quad] |= MASK_START_E;
                                start_in_row |= ANY_START(quad);
                                break;
                        }
                    }
                    break;
                case MASK_EXISTS_NW_CORNER:
                    if (_filled) {
                        switch ((z_nw << 4) | (z_ne << 2) | z_sw) {  // config
                            case  1:  // 001
                            case  5:  // 011
                            case  9:  // 021
                            case 10:  // 022
                            case 16:  // 100
                            case 17:  // 101
                            case 25:  // 121
                            case 26:  // 122
                            case 32:  // 200
                            case 33:  // 201
                            case 37:  // 211
                            case 41:  // 221
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                            case  2:  // 002
                            case  6:  // 012
                            case 18:  // 102
                            case 24:  // 120
                            case 36:  // 210
                            case 40:  // 220
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case  4:  // 010
                            case  8:  // 020
                            case 34:  // 202
                            case 38:  // 212
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 20:  // 110
                            case 22:  // 112
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 21:  // 111
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                start_in_row |= ANY_START(quad);
                                break;
                        }
                    }
                    else {  // !_filled NW corner.
                        switch ((z_nw << 2) | (z_ne << 1) | z_sw) {  // config
                            case 1:  // 001
                            case 5:  // 101
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 2:  // 010
                            case 3:  // 011
                                if (BOUNDARY_N(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                    start_in_row = true;
                                }
                                break;
                            case 4:  // 100
                            case 6:  // 110
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                        }
                    }
                    break;
                case MASK_EXISTS_NE_CORNER:
                    if (_filled) {
                        switch ((z_nw << 4) | (z_ne << 2) | z_se) {  // config
                            case  1:  // 001
                            case  2:  // 002
                            case  5:  // 011
                            case  6:  // 012
                            case 10:  // 022
                            case 16:  // 100
                            case 26:  // 122
                            case 32:  // 200
                            case 36:  // 210
                            case 37:  // 211
                            case 40:  // 220
                            case 41:  // 221
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case  4:  // 010
                            case 38:  // 212
                                _cache[quad] |= MASK_START_N;
                                start_in_row = true;
                                break;
                            case  8:  // 020
                            case 34:  // 202
                                _cache[quad] |= MASK_START_E;
                                _cache[quad] |= MASK_START_N;
                                start_in_row = true;
                                break;
                            case  9:  // 021
                            case 17:  // 101
                            case 18:  // 102
                            case 24:  // 120
                            case 25:  // 121
                            case 33:  // 201
                                _cache[quad] |= MASK_START_CORNER;
                                _cache[quad] |= MASK_START_E;
                                start_in_row = true;
                                break;
                            case 20:  // 110
                            case 21:  // 111
                            case 22:  // 112
                                if (BOUNDARY_N(quad) && !START_HOLE_N(quad-1) &&
                                    j % _y_chunk_size > 0 && j != _ny-1 && i % _x_chunk_size > 1)
                                    _cache[quad] |= MASK_START_HOLE_N;
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                        }
                    }
                    else {  // !_filled NE corner.
                        switch ((z_nw << 2) | (z_ne << 1) | z_se) {  // config
                            case 1:  // 001
                                if (BOUNDARY_E(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                    start_in_row = true;
                                }
                                break;
                            case 2:  // 010
                                if (BOUNDARY_N(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                else if (!BOUNDARY_E(quad))
                                    _cache[quad] |= MASK_START_N;
                                start_in_row |= ANY_START(quad);
                                break;
                            case 3:  // 011
                                if (BOUNDARY_N(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_N;
                                    start_in_row = true;
                                }
                                break;
                            case 4:  // 100
                            case 6:  // 110
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 5:  // 101
                                if (BOUNDARY_E(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                else if (!BOUNDARY_N(quad))
                                    _cache[quad] |= MASK_START_E;
                                start_in_row |= ANY_START(quad);
                                break;
                        }
                    }
                    break;
                case MASK_EXISTS_SW_CORNER:
                    if (_filled) {
                        switch ((z_nw << 4) | (z_sw << 2) | z_se) {  // config
                            case  1:  // 001
                            case  2:  // 002
                            case 40:  // 220
                            case 41:  // 221
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case  4:  // 010
                            case  5:  // 011
                            case  6:  // 012
                            case  8:  // 020
                            case  9:  // 021
                            case 18:  // 102
                            case 24:  // 120
                            case 33:  // 201
                            case 34:  // 202
                            case 36:  // 210
                            case 37:  // 211
                            case 38:  // 212
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                start_in_row |= ANY_START(quad);
                                break;
                            case 10:  // 022
                            case 16:  // 100
                            case 26:  // 122
                            case 32:  // 200
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                            case 17:  // 101
                            case 25:  // 121
                                if (BOUNDARY_S(quad)) _cache[quad] |= MASK_START_BOUNDARY_S;
                                if (BOUNDARY_W(quad)) _cache[quad] |= MASK_START_BOUNDARY_W;
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 20:  // 110
                            case 21:  // 111
                            case 22:  // 112
                                if (BOUNDARY_S(quad))
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                else
                                    _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                        }
                    }
                    else {  // !_filled SW corner.
                        switch ((z_nw << 2) | (z_sw << 1) | z_se) {  // config
                            case 1:  // 001
                            case 3:  // 011
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                            case 2:  // 010
                            case 6:  // 110
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case 4:  // 100
                            case 5:  // 101
                                if (BOUNDARY_W(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_W;
                                    start_in_row = true;
                                }
                                break;
                        }
                    }
                    break;
                case MASK_EXISTS_SE_CORNER:
                    if (_filled) {
                        switch ((z_ne << 4) | (z_sw << 2) | z_se) {  // config
                            case  1:  // 001
                            case  2:  // 002
                            case  4:  // 010
                            case  5:  // 011
                            case  6:  // 012
                            case  8:  // 020
                            case  9:  // 021
                            case 17:  // 101
                            case 18:  // 102
                            case 20:  // 110
                            case 21:  // 111
                            case 22:  // 112
                            case 24:  // 120
                            case 25:  // 121
                            case 33:  // 201
                            case 34:  // 202
                            case 36:  // 210
                            case 37:  // 211
                            case 38:  // 212
                            case 40:  // 220
                            case 41:  // 221
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case 10:  // 022
                            case 16:  // 100
                            case 26:  // 122
                            case 32:  // 200
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                        }
                    }
                    else {  // !_filled SE corner.
                        switch ((z_ne << 2) | (z_sw << 1) | z_se) {  // config
                            case 1:  // 001
                            case 3:  // 011
                                if (BOUNDARY_E(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_E;
                                    start_in_row = true;
                                }
                                break;
                            case 2:  // 010
                            case 6:  // 110
                                if (BOUNDARY_S(quad)) {
                                    _cache[quad] |= MASK_START_BOUNDARY_S;
                                    start_in_row = true;
                                }
                                break;
                            case 4:  // 100
                            case 5:  // 101
                                _cache[quad] |= MASK_START_CORNER;
                                start_in_row = true;
                                break;
                        }
                    }
                    break;
            }

            z_nw = z_ne;
            z_sw = z_se;
        } // i-loop.

        if (start_in_row)
            j_final_start = j;
        else if (j > 0)
            _cache[chunk_istart + j*_nx] |= MASK_NO_STARTS_IN_ROW;
    } // j-loop.

    if (j_final_start < jend)
        _cache[chunk_istart + (j_final_start+1)*_nx] |= MASK_NO_MORE_STARTS;
}

template <typename Derived>
void BaseContourGenerator<Derived>::interp(
    index_t point0, index_t point1, bool is_upper, double*& points) const
{
    auto frac = get_interp_fraction(
        get_point_z(point0), get_point_z(point1), is_upper ? _upper_level : _lower_level);

    assert(frac >= 0.0 && frac <= 1.0 && "Interp fraction out of bounds");

    *points++ = get_point_x(point0)*frac + get_point_x(point1)*(1.0 - frac);
    *points++ = get_point_y(point0)*frac + get_point_y(point1)*(1.0 - frac);
}

template <typename Derived>
void BaseContourGenerator<Derived>::interp(
    index_t point0, double x1, double y1, double z1, bool is_upper, double*& points) const
{
    auto frac = get_interp_fraction(
        get_point_z(point0), z1, is_upper ? _upper_level : _lower_level);

    assert(frac >= 0.0 && frac <= 1.0 && "Interp fraction out of bounds");

    *points++ = get_point_x(point0)*frac + x1*(1.0 - frac);
    *points++ = get_point_y(point0)*frac + y1*(1.0 - frac);
}

template <typename Derived>
bool BaseContourGenerator<Derived>::is_filled() const
{
    return _filled;
}

template <typename Derived>
bool BaseContourGenerator<Derived>::is_point_in_chunk(index_t point, const ChunkLocal& local) const
{
    return is_quad_in_bounds(point, local.istart-1, local.iend, local.jstart-1, local.jend);
}

template <typename Derived>
bool BaseContourGenerator<Derived>::is_quad_in_bounds(
    index_t quad, index_t istart, index_t iend, index_t jstart, index_t jend) const
{
    return (quad % _nx >= istart && quad % _nx <= iend &&
            quad / _nx >= jstart && quad / _nx <= jend);
}

template <typename Derived>
bool BaseContourGenerator<Derived>::is_quad_in_chunk(index_t quad, const ChunkLocal& local) const
{
    return is_quad_in_bounds(quad, local.istart, local.iend, local.jstart, local.jend);
}

template <typename Derived>
void BaseContourGenerator<Derived>::line(const Location& start_location, ChunkLocal& local)
{
    // start_location.on_boundary indicates starts (and therefore also finishes)

    assert(is_quad_in_chunk(start_location.quad, local));

    Location location = start_location;
    count_t point_count = 0;

    // Insert nan if required before start of new line.
    if (_nan_separated and local.pass > 0 && local.line_count > 0) {
        *local.points.current++ = Util::nan;
        *local.points.current++ = Util::nan;
    }

    // finished == true indicates closed line loop.
    bool finished = follow_interior(location, start_location, local, point_count);

    if (local.pass > 0) {
        assert(local.line_offsets.current == local.line_offsets.start + local.line_count);
        *local.line_offsets.current++ = local.total_point_count;
    }

    if (local.pass == 0 && !start_location.on_boundary && !finished)
        // An internal start that isn't a line loop is part of a line strip that starts on a
        // boundary and will be traced later.  Do not count it as a valid start in pass 0 and remove
        // the first point or it will be duplicated by the correct boundary-started line later.
        point_count--;
    else
        local.line_count++;

    local.total_point_count += point_count;
}

template <typename Derived>
py::sequence BaseContourGenerator<Derived>::lines(double level)
{
    _filled = false;
    _lower_level = _upper_level = level;

    _identify_holes = false;
    _output_chunked = !(_line_type == LineType::Separate || _line_type == LineType::SeparateCode);
    _direct_points = _output_chunked;
    _direct_line_offsets = (_line_type == LineType::ChunkCombinedOffset);
    _direct_outer_offsets = false;
    _outer_offsets_into_points = false;
    _return_list_count = (_line_type == LineType::Separate ||
                          _line_type == LineType::ChunkCombinedNan) ? 1 : 2;
    _nan_separated = (_line_type == LineType::ChunkCombinedNan);

    if (_nan_separated)
        Util::ensure_nan_loaded();

    return march_wrapper();
}

template <typename Derived>
void BaseContourGenerator<Derived>::march_chunk(
    ChunkLocal& local, std::vector<py::list>& return_lists)
{
    for (local.pass = 0; local.pass < 2; ++local.pass) {
        bool ignore_holes = (_identify_holes && local.pass == 1);

        index_t j_final_start = local.jstart;
        for (index_t j = local.jstart; j <= local.jend; ++j) {
            index_t quad = local.istart + j*_nx;

            if (NO_MORE_STARTS(quad))
                break;

            if (NO_STARTS_IN_ROW(quad))
                continue;

            // Want to count number of starts in this row, so store how many starts at start of row.
            auto prev_start_count =
                (_identify_holes ? local.line_count - local.hole_count : local.line_count);

            for (index_t i = local.istart; i <= local.iend; ++i, ++quad) {
                if (!ANY_START(quad))
                    continue;

                assert(EXISTS_ANY(quad));

                if (_filled) {
                    if (START_BOUNDARY_S(quad))
                        closed_line_wrapper(Location(quad, 1, _nx, Z_SW == 2, true), Outer, local);

                    if (START_BOUNDARY_W(quad))
                        closed_line_wrapper(Location(quad, -_nx, 1, Z_NW == 2, true), Outer, local);

                    if (START_CORNER(quad)) {
                        switch (EXISTS_ANY_CORNER(quad)) {
                            case MASK_EXISTS_NE_CORNER:
                                closed_line_wrapper(
                                    Location(quad, -_nx+1, _nx+1, Z_NW == 2, true), Outer, local);
                                break;
                            case MASK_EXISTS_NW_CORNER:
                                closed_line_wrapper(
                                    Location(quad, _nx+1, _nx-1, Z_SW == 2, true), Outer, local);
                                break;
                            case MASK_EXISTS_SE_CORNER:
                                closed_line_wrapper(
                                    Location(quad, -_nx-1, -_nx+1, Z_NE == 2, true), Outer, local);
                                break;
                            default:
                                assert(EXISTS_SW_CORNER(quad));
                                if (!ignore_holes)
                                    closed_line_wrapper(
                                        Location(quad, _nx-1, -_nx-1, false, true), Hole, local);
                                break;
                        }
                    }

                    if (START_N(quad))
                        closed_line_wrapper(Location(quad, -_nx, 1, Z_NW > 0, false), Outer, local);

                    if (ignore_holes)
                        continue;

                    if (START_E(quad))
                        closed_line_wrapper(Location(quad, -1, -_nx, Z_NE > 0, false), Hole, local);

                    if (START_HOLE_N(quad))
                        closed_line_wrapper(Location(quad, -1, -_nx, false, true), Hole, local);
                }
                else {  // !_filled
                    if (START_BOUNDARY_S(quad))
                        line(Location(quad, _nx, -1, false, true), local);

                    if (START_BOUNDARY_W(quad))
                        line(Location(quad, 1, _nx, false, true), local);

                    if (START_BOUNDARY_E(quad))
                        line(Location(quad, -1, -_nx, false, true), local);

                    if (START_BOUNDARY_N(quad))
                        line(Location(quad, -_nx, 1, false, true), local);

                    if (START_E(quad))
                        line(Location(quad, -1, -_nx, false, false), local);

                    if (START_N(quad))
                        line(Location(quad, -_nx, 1, false, false), local);

                    if (START_CORNER(quad)) {
                        index_t forward, left;
                        switch (EXISTS_ANY_CORNER(quad)) {
                            case MASK_EXISTS_NE_CORNER:
                                forward = _nx+1;
                                left = _nx-1;
                                break;
                            case MASK_EXISTS_NW_CORNER:
                                forward = _nx-1;
                                left = -_nx-1;
                                break;
                            case MASK_EXISTS_SE_CORNER:
                                forward = -_nx+1;
                                left = _nx+1;
                                break;
                            default:
                                assert(EXISTS_SW_CORNER(quad));
                                forward = -_nx-1;
                                left = -_nx+1;
                                break;
                        }
                        line(Location(quad, forward, left, false, true), local);
                    }
                } // _filled
            } // i

            // Number of starts at end of row.
            auto start_count =
                (_identify_holes ? local.line_count - local.hole_count : local.line_count);
            if (start_count > prev_start_count)
                j_final_start = j;
            else
                _cache[local.istart + j*_nx] |= MASK_NO_STARTS_IN_ROW;
        } // j

        if (j_final_start < local.jend)
            _cache[local.istart + (j_final_start+1)*_nx] |= MASK_NO_MORE_STARTS;

        if (_nan_separated && local.line_count > 1) {
            // If _nan_separated, each line after the first has an extra nan to separate it from the
            // previous line's points. If we were returning line offsets to the caller then this
            // would need to occur in line() where the line_count is incremented. But as we are not
            // returning line offsets it is faster just to add the extra here all at once.
            local.total_point_count += local.line_count - 1;
        }

        if (local.pass == 0) {
            if (local.total_point_count == 0) {
                local.points.clear();
                local.line_offsets.clear();
                local.outer_offsets.clear();
                break;  // Do not need pass 1.
            }

            // Create arrays for points, line_offsets and optionally outer_offsets.  Arrays may be
            // either C++ vectors or Python NumPy arrays.  Want to group creation of the latter as
            // threaded code needs to lock creation of these to limit access to a single thread.
            if (_direct_points || _direct_line_offsets || _direct_outer_offsets) {
                typename Derived::Lock lock(static_cast<Derived&>(*this));

                // Strictly speaking adding the NumPy arrays to return_lists does not need to be
                // within the lock.
                if (_direct_points) {
                    return_lists[0][local.chunk] =
                        local.points.create_python(local.total_point_count, 2);
                }
                if (_direct_line_offsets) {
                    return_lists[1][local.chunk] =
                        local.line_offsets.create_python(local.line_count + 1);
                }
                if (_direct_outer_offsets) {
                    return_lists[2][local.chunk] =
                        local.outer_offsets.create_python(local.line_count - local.hole_count + 1);
                }
            }

            if (!_direct_points)
                local.points.create_cpp(2*local.total_point_count);

            if (!_direct_line_offsets)
                local.line_offsets.create_cpp(local.line_count + 1);

            if (!_direct_outer_offsets) {
                if (_identify_holes)
                    local.outer_offsets.create_cpp( local.line_count - local.hole_count + 1);
                else
                    local.outer_offsets.clear();
            }

            // Reset counts for pass 1.
            local.total_point_count = 0;
            local.line_count = 0;
            local.hole_count = 0;
        }
    } // pass

    // Set final line and outer offsets.
    if (local.line_count > 0) {
        *local.line_offsets.current++ = local.total_point_count;

        if (_identify_holes) {
            if (_outer_offsets_into_points)
                *local.outer_offsets.current++ = local.total_point_count;
            else
                *local.outer_offsets.current++ = local.line_count;
        }
    }

    // Throw exception if the two passes returned different number of points, lines, etc.
    check_consistent_counts(local);

    if (local.total_point_count == 0) {
        if (_output_chunked) {
            typename Derived::Lock lock(static_cast<Derived&>(*this));
            for (auto& list : return_lists)
                list[local.chunk] = py::none();
        }
    }
    else if (_filled)
        static_cast<Derived*>(this)->export_filled(local, return_lists);
    else
        static_cast<Derived*>(this)->export_lines(local, return_lists);
}

template <typename Derived>
py::sequence BaseContourGenerator<Derived>::march_wrapper()
{
    index_t list_len = _n_chunks;
    if ((_filled && (_fill_type == FillType::OuterCode|| _fill_type == FillType::OuterOffset)) ||
        (!_filled && (_line_type == LineType::Separate || _line_type == LineType::SeparateCode)))
        list_len = 0;

    // Prepare lists to return to python.
    std::vector<py::list> return_lists;
    return_lists.reserve(_return_list_count);
    for (decltype(_return_list_count) i = 0; i < _return_list_count; ++i)
        return_lists.emplace_back(list_len);

    static_cast<Derived*>(this)->march(return_lists);

    // Return to python objects.
    if (_return_list_count == 1) {
        assert(!_filled);
        if (_line_type == LineType::Separate)
            return return_lists[0];
        else {
            assert(_line_type == LineType::ChunkCombinedNan);
            return py::make_tuple(return_lists[0]);
        }
    }
    else if (_return_list_count == 2)
        return py::make_tuple(return_lists[0], return_lists[1]);
    else {
        assert(_return_list_count == 3);
        return py::make_tuple(return_lists[0], return_lists[1], return_lists[2]);
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::move_to_next_boundary_edge(
    index_t& quad, index_t& forward, index_t& left) const
{
    // edge == 0 for E edge (facing N), forward = +_nx
    //         2 for S edge (facing E), forward = +1
    //         4 for W edge (facing S), forward = -_nx
    //         6 for N edge (facing W), forward = -1
    //         1 for SE edge (NW corner) from SW facing NE, forward = +_nx+1
    //         3 for SW edge (NE corner) from NW facing SE, forward = -_nx+1
    //         5 for NW edge (SE corner) from NE facing SW, forward = -_nx-1
    //         7 for NE edge (SW corner) from SE facing NW, forward = +_nx-1
    int edge = 0;

    // Need index of quad that is the same as the end point, i.e. quad to SW of end point, as it is
    // this point which we need to find the next available boundary of, looking clockwise.
    if (forward > 0) {
        if (forward == _nx) {
            assert(left == -1);
            // W edge facing N, no change to quad or edge.
        }
        else if (left == _nx) {
            assert(forward == 1);
            quad -= _nx;  // S edge facing E.
            edge = 2;
        }
        else if (EXISTS_SW_CORNER(quad)) {
            assert(forward == _nx-1 && left == -_nx-1);
            quad -= 1;
            edge = 7;
        }
        else {
            assert(EXISTS_NW_CORNER(quad) && forward == _nx+1 && _nx-1);
            // quad unchanged.
            edge = 1;
        }
    }
    else {  // forward < 0
        if (forward == -_nx) {
            assert(left == 1);
            quad -= _nx+1;  // W edge facing S.
            edge = 4;
        }
        else if (left == -_nx) {
            assert(forward == -1);
            quad -= 1;  // N edge facing W.
            edge = 6;
        }
        else if (EXISTS_NE_CORNER(quad)) {
            assert(forward == -_nx+1 && left == _nx+1);
            quad -= _nx;
            edge = 3;
        }
        else {
            assert(EXISTS_SE_CORNER(quad) && forward == -_nx-1 && left == -_nx+1);
            quad -= _nx+1;
            edge = 5;
        }
    }

    // If _corner_mask not set, only need to consider odd edge in loop below.
    if (!_corner_mask)
        ++edge;

    while (true) {
        // Look at possible edges that leave NE point of quad.
        // If something is wrong here or in the setup of the boundary flags, can end up with an
        // infinite loop!
        switch (edge) {
            case 0:
                // Is there an edge to follow towards SW?
                if (EXISTS_SE_CORNER(quad)) {  // Equivalent to BOUNDARY_NE.
                    // quad unchanged.
                    forward = -_nx-1;
                    left = -_nx+1;
                    return;
                }
                break;
            case 1:
                // Is there an edge to follow towards W?
                if (BOUNDARY_N(quad)) {
                    // quad unchanged.
                    forward = -1;
                    left = -_nx;
                    return;
                }
                break;
            case 2:
                // Is there an edge to follow towards NW?
                if (EXISTS_SW_CORNER(quad+_nx)) {  // Equivalent to BOUNDARY_NE.
                    quad += _nx;
                    forward = _nx-1;
                    left = -_nx-1;
                    return;
                }
                break;
            case 3:
                // Is there an edge to follow towards N?
                if (BOUNDARY_E(quad+_nx)) {  // Really a BOUNDARY_W check.
                    quad += _nx;
                    forward = _nx;
                    left = -1;
                    return;
                }
                break;
            case 4:
                // Is there an edge to follow towards NE?
                if (EXISTS_NW_CORNER(quad+_nx+1)) {  // Equivalent to BOUNDARY_SE.
                    quad += _nx+1;
                    forward = _nx+1;
                    left = _nx-1;
                    return;
                }
                break;
            case 5:
                // Is there an edge to follow towards E?
                if (BOUNDARY_N(quad+1)) {  // Really a BOUNDARY_S check
                    quad += _nx+1;
                    forward = 1;
                    left = _nx;
                    return;
                }
                break;
            case 6:
                // Is there an edge to follow towards SE?
                if (EXISTS_NE_CORNER(quad+1)) {  // Equivalent to BOUNDARY_SW.
                    quad += 1;
                    forward = -_nx+1;
                    left = _nx+1;
                    return;
                }
                break;
            case 7:
                // Is there an edge to follow towards S?
                if (BOUNDARY_E(quad)) {
                    quad += 1;
                    forward = -_nx;
                    left = 1;
                    return;
                }
                break;
            default:
                assert(0 && "Invalid edge index");
                break;
        }

        edge = _corner_mask ? (edge + 1) % 8 : (edge + 2) % 8;
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::set_look_flags(index_t hole_start_quad)
{
    assert(_identify_holes);

    // The only possible hole starts are START_E (from E to N), START_HOLE_N (on N boundary, E to W)
    // and START_CORNER for SW corner (on boundary, SE to NW).
    assert(hole_start_quad >= 0 && hole_start_quad < _n);
    assert(EXISTS_N_EDGE(hole_start_quad) || EXISTS_SW_CORNER(hole_start_quad));
    assert(!LOOK_S(hole_start_quad) && "Look S already set");

    _cache[hole_start_quad] |= MASK_LOOK_S;

    // Walk S until find place to mark corresponding look N.
    auto quad = hole_start_quad;

    while (true) {
        assert(quad >= 0 && quad < _n);
        assert(EXISTS_N_EDGE(quad) || (quad == hole_start_quad && EXISTS_SW_CORNER(quad)));

        if (BOUNDARY_S(quad) || EXISTS_NE_CORNER(quad) || EXISTS_NW_CORNER(quad) || Z_SE != 1) {
            assert(!LOOK_N(quad) && "Look N already set");
            _cache[quad] |= MASK_LOOK_N;
            break;
        }

        quad -= _nx;
    }
}

template <typename Derived>
bool BaseContourGenerator<Derived>::supports_fill_type(FillType fill_type)
{
    switch (fill_type) {
        case FillType::OuterCode:
        case FillType::OuterOffset:
        case FillType::ChunkCombinedCode:
        case FillType::ChunkCombinedOffset:
        case FillType::ChunkCombinedCodeOffset:
        case FillType::ChunkCombinedOffsetOffset:
            return true;
        default:
            return false;
    }
}

template <typename Derived>
bool BaseContourGenerator<Derived>::supports_line_type(LineType line_type)
{
    switch (line_type) {
        case LineType::Separate:
        case LineType::SeparateCode:
        case LineType::ChunkCombinedCode:
        case LineType::ChunkCombinedOffset:
        case LineType::ChunkCombinedNan:
            return true;
        default:
            return false;
    }
}

template <typename Derived>
void BaseContourGenerator<Derived>::write_cache() const
{
    std::cout << "---------- Cache ----------" << std::endl;
    index_t ny = _n / _nx;
    for (index_t j = ny-1; j >= 0; --j) {
        std::cout << "j=" << j << " ";
        for (index_t i = 0; i < _nx; ++i) {
            index_t quad = i + j*_nx;
            write_cache_quad(quad);
        }
        std::cout << std::endl;
    }
    std::cout << "    ";
    for (index_t i = 0; i < _nx; ++i)
        std::cout << "i=" << i << "           ";
    std::cout << std::endl;
    std::cout << "---------------------------" << std::endl;
}

template <typename Derived>
void BaseContourGenerator<Derived>::write_cache_quad(index_t quad) const
{
    assert(quad >= 0 && quad < _n && "quad index out of bounds");
    std::cout << (NO_MORE_STARTS(quad) ? 'x' :
                    (NO_STARTS_IN_ROW(quad) ? 'i' : '.'));
    std::cout << (EXISTS_QUAD(quad) ? "Q_" :
                   (EXISTS_NW_CORNER(quad) ? "NW" :
                     (EXISTS_NE_CORNER(quad) ? "NE" :
                       (EXISTS_SW_CORNER(quad) ? "SW" :
                         (EXISTS_SE_CORNER(quad) ? "SE" : "..")))));
    std::cout << (BOUNDARY_N(quad) && BOUNDARY_E(quad) ? 'b' : (
                    BOUNDARY_N(quad) ? 'n' : (BOUNDARY_E(quad) ? 'e' : '.')));
    std::cout << Z_LEVEL(quad);
    std::cout << ((_cache[quad] & MASK_MIDDLE) >> 2);
    std::cout << (START_BOUNDARY_S(quad) ? 's' : '.');
    std::cout << (START_BOUNDARY_W(quad) ? 'w' : '.');
    if (!_filled) {
        std::cout << (START_BOUNDARY_E(quad) ? 'e' : '.');
        std::cout << (START_BOUNDARY_N(quad) ? 'n' : '.');
    }
    std::cout << (START_E(quad) ? 'E' : '.');
    std::cout << (START_N(quad) ? 'N' : '.');
    if (_filled)
        std::cout << (START_HOLE_N(quad) ? 'h' : '.');
    std::cout << (START_CORNER(quad) ? 'c' : '.');
    if (_filled)
        std::cout << (LOOK_N(quad) && LOOK_S(quad) ? 'B' :
            (LOOK_N(quad) ? '^' : (LOOK_S(quad) ? 'v' : '.')));
    std::cout << ' ';
}

template <typename Derived>
typename BaseContourGenerator<Derived>::ZLevel BaseContourGenerator<Derived>::z_to_zlevel(
    double z_value) const
{
    return (_filled && z_value > _upper_level) ? 2 : (z_value > _lower_level ? 1 : 0);
}

} // namespace contourpy

#endif // CONTOURPY_BASE_IMPL_H
