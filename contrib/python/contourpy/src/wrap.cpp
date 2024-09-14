#include "base_impl.h"
#include "contour_generator.h"
#include "fill_type.h"
#include "line_type.h"
#include "mpl2005.h"
#include "mpl2014.h"
#include "serial.h"
#include "threaded.h"
#include "util.h"
#include "z_interp.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

static contourpy::LineType mpl20xx_line_type = contourpy::LineType::SeparateCode;
static contourpy::FillType mpl20xx_fill_type = contourpy::FillType::OuterCode;

PYBIND11_MODULE(_contourpy, m, py::mod_gil_not_used()) {
    m.doc() =
        "C++11 extension module wrapped using `pybind11`_.\n\n"
        "It should not be necessary to access classes and functions in this extension module "
        "directly. Instead, :func:`~contourpy.contour_generator` should be used to create "
        ":class:`~.ContourGenerator` objects, and the enums "
        "(:class:`~.FillType`, :class:`~.LineType` and "
        ":class:`~.ZInterp`) and :func:`.max_threads` function are all available "
        "in the :mod:`contourpy` module.";

    m.attr("__version__") = MACRO_STRINGIFY(CONTOURPY_VERSION);

    // asserts are enabled if NDEBUG is 0.
    m.attr("NDEBUG") =
#ifdef NDEBUG
        1;
#else
        0;
#endif

    py::enum_<contourpy::FillType>(m, "FillType",
        "Enum used for ``fill_type`` keyword argument in :func:`~.contour_generator`.\n\n"
        "This controls the format of filled contour data returned from "
        ":meth:`~.ContourGenerator.filled`.")
        .value("OuterCode", contourpy::FillType::OuterCode)
        .value("OuterOffset", contourpy::FillType::OuterOffset)
        .value("ChunkCombinedCode", contourpy::FillType::ChunkCombinedCode)
        .value("ChunkCombinedOffset", contourpy::FillType::ChunkCombinedOffset)
        .value("ChunkCombinedCodeOffset", contourpy::FillType::ChunkCombinedCodeOffset)
        .value("ChunkCombinedOffsetOffset", contourpy::FillType::ChunkCombinedOffsetOffset)
        .export_values();

    py::enum_<contourpy::LineType>(m, "LineType",
        "Enum used for ``line_type`` keyword argument in :func:`~.contour_generator`.\n\n"
        "This controls the format of contour line data returned from "
        ":meth:`~.ContourGenerator.lines`.\n\n"
        "``LineType.ChunkCombinedNan`` added in version 1.2.0")
        .value("Separate", contourpy::LineType::Separate)
        .value("SeparateCode", contourpy::LineType::SeparateCode)
        .value("ChunkCombinedCode", contourpy::LineType::ChunkCombinedCode)
        .value("ChunkCombinedOffset", contourpy::LineType::ChunkCombinedOffset)
        .value("ChunkCombinedNan", contourpy::LineType::ChunkCombinedNan)
        .export_values();

    py::enum_<contourpy::ZInterp>(m, "ZInterp",
        "Enum used for ``z_interp`` keyword argument in :func:`~.contour_generator`\n\n"
        "This controls the interpolation used on ``z`` values to determine where contour lines "
        "intersect the edges of grid quads, and ``z`` values at quad centres.")
        .value("Linear", contourpy::ZInterp::Linear)
        .value("Log", contourpy::ZInterp::Log)
        .export_values();

    m.def("max_threads", &contourpy::Util::get_max_threads,
        "Return the maximum number of threads, obtained from "
        "``std::thread::hardware_concurrency()``.\n\n"
        "This is the number of threads used by a multithreaded ContourGenerator if the kwarg "
        "``threads=0`` is passed to :func:`~.contour_generator`.");

    const char* chunk_count_doc = "Return tuple of (y, x) chunk counts.";
    const char* chunk_size_doc = "Return tuple of (y, x) chunk sizes.";
    const char* corner_mask_doc = "Return whether ``corner_mask`` is set or not.";
    const char* create_contour_doc =
        "Synonym for :meth:`~.ContourGenerator.lines` to provide backward compatibility "
        "with Matplotlib.";
    const char* create_filled_contour_doc =
        "Synonym for :meth:`~.ContourGenerator.filled` to provide backward compatibility "
        "with Matplotlib.";
    const char* default_fill_type_doc =
        "Return the default :class:`~.FillType` used by this algorithm.";
    const char* default_line_type_doc =
        "Return the default :class:`~.LineType` used by this algorithm.";
    const char* fill_type_doc = "Return the :class:`~.FillType`.";
    const char* filled_doc =
        "Calculate and return filled contours between two levels.\n\n"
        "Args:\n"
        "    lower_level (float): Lower z-level of the filled contours, cannot be ``np.nan``.\n"
        "    upper_level (float): Upper z-level of the filled contours, cannot be ``np.nan``.\n"
        "Return:\n"
        "    Filled contour polygons as nested sequences of numpy arrays. The exact format is "
        "determined by the :attr:`~.ContourGenerator.fill_type` used by the ``ContourGenerator`` "
        "and the options are explained at :ref:`fill_type`.\n\n"
        "Raises a ``ValueError`` if ``lower_level >= upper_level`` or if\n"
        "``lower_level`` or ``upper_level`` are ``np.nan``.\n\n"
        "To return filled contours below a ``level`` use ``filled(-np.inf, level)``.\n"
        "To return filled contours above a ``level`` use ``filled(level, np.inf)``";
    const char* line_type_doc = "Return the :class:`~.LineType`.";
    const char* lines_doc =
        "Calculate and return contour lines at a particular level.\n\n"
        "Args:\n"
        "    level (float): z-level to calculate contours at.\n\n"
        "Return:\n"
        "    Contour lines (open line strips and closed line loops) as nested sequences of "
        "numpy arrays. The exact format is determined by the :attr:`~.ContourGenerator.line_type` "
        "used by the ``ContourGenerator`` and the options are explained at :ref:`line_type`.\n\n"
        "``level`` may be ``np.nan``, ``np.inf`` or ``-np.inf``; they all return the same result "
        "which is an empty line set.";
    const char* multi_filled_doc =
        "Calculate and return filled contours between multiple pairs of adjacent levels.\n\n"
        "Args:\n"
        "    levels (array-like of floats): z-levels to calculate filled contours between. "
        "There must be at least 2 levels, they cannot be NaN, and each level must be larger than "
        "the previous level.\n\n"
        "Return:\n"
        "    List of filled contours, one per pair of levels. The length of the returned list is "
        "one less than ``len(levels)``. The format of returned contours is determined by the "
        "``fill_type`` used by the ``ContourGenerator`` and the options are explained at "
        ":ref:`fill_type`.\n\n"
        "Raises a ``ValueError`` if ``levels`` are not increasing, or contain a NaN, or there are "
        "fewer than 2 ``levels``.\n\n"
        "Calling:\n\n"
        ".. code-block:: python\n\n"
        "    ret = cont_gen.multi_filled(levels)\n\n"
        "is equivalent to:\n\n"
        ".. code-block:: python\n\n"
        "    ret = [cont_gen.filled(lower, upper) for lower, upper in zip(levels[:-1], levels[1:])]\n\n"
        ".. versionadded:: 1.3.0";
    const char* multi_lines_doc =
        "Calculate and return contour lines at multiple levels.\n\n"
        "Args:\n"
        "    levels (array-like of floats): z-levels to calculate contours at.\n\n"
        "Return:\n"
        "    List of contour lines, one per level. The format of returned lines is determined by "
        "the ``line_type`` used by the ``ContourGenerator`` and the options are explained at "
        ":ref:`line_type`.\n\n"
        "Calling:\n\n"
        ".. code-block:: python\n\n"
        "    ret = cont_gen.multi_lines(levels)\n\n"
        "is equivalent to:\n\n"
        ".. code-block:: python\n\n"
        "    ret = [cont_gen.lines(level) for level in levels]\n\n"
        ".. versionadded:: 1.3.0";
    const char* quad_as_tri_doc = "Return whether ``quad_as_tri`` is set or not.";
    const char* supports_corner_mask_doc =
        "Return whether this algorithm supports ``corner_mask``.";
    const char* supports_fill_type_doc =
        "Return whether this algorithm supports a particular :class:`~.FillType`.";
    const char* supports_line_type_doc =
        "Return whether this algorithm supports a particular :class:`~.LineType`.";
    const char* supports_quad_as_tri_doc =
        "Return whether this algorithm supports ``quad_as_tri``.";
    const char* supports_threads_doc =
        "Return whether this algorithm supports the use of threads.";
    const char* supports_z_interp_doc =
        "Return whether this algorithm supports ``z_interp`` values other than ``ZInterp.Linear`` "
        "which all support.";
    const char* thread_count_doc = "Return the number of threads used.";
    const char* z_interp_doc = "Return the ``ZInterp``.";

    py::class_<contourpy::ContourGenerator>(m, "ContourGenerator",
        "Abstract base class for contour generator classes, defining the interface that they all "
        "implement.")
        .def("create_contour", &contourpy::ContourGenerator::lines, create_contour_doc, "level"_a)
        .def("create_filled_contour", &contourpy::ContourGenerator::filled,
             create_filled_contour_doc, "lower_level"_a, "upper_level"_a)
        .def("filled", &contourpy::ContourGenerator::filled, filled_doc,
             "lower_level"_a, "upper_level"_a)
        .def("lines", &contourpy::ContourGenerator::lines, lines_doc, "level"_a)
        .def("multi_filled", &contourpy::ContourGenerator::multi_filled, multi_filled_doc,
             "levels"_a)
        .def("multi_lines", &contourpy::ContourGenerator::multi_lines, multi_lines_doc, "levels"_a)
        .def_property_readonly(
            "chunk_count", [](py::object /* self */) {return py::make_tuple(1, 1);},
            chunk_count_doc)
        .def_property_readonly(
            "chunk_size", [](py::object /* self */) {return py::make_tuple(1, 1);}, chunk_size_doc)
        .def_property_readonly(
            "corner_mask", [](py::object /* self */) {return false;}, corner_mask_doc)
        .def_property_readonly(
            "fill_type", [](py::object /* self */) {return contourpy::FillType::OuterOffset;},
            fill_type_doc)
        .def_property_readonly(
            "line_type", [](py::object /* self */) {return contourpy::LineType::Separate;},
            line_type_doc)
        .def_property_readonly(
            "quad_as_tri", [](py::object /* self */) {return false;}, quad_as_tri_doc)
        .def_property_readonly(
            "thread_count", [](py::object /* self */) {return 1;}, thread_count_doc)
        .def_property_readonly(
            "z_interp", [](py::object /* self */) {return contourpy::ZInterp::Linear;},
            z_interp_doc)
        .def_property_readonly_static(
            "default_fill_type",
            [](py::object /* self */) {return contourpy::FillType::OuterOffset;},
            default_fill_type_doc)
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return contourpy::LineType::Separate;},
            default_line_type_doc)
        .def_static("supports_corner_mask", []() {return false;}, supports_corner_mask_doc)
        .def_static(
            "supports_fill_type", [](contourpy::FillType /* fill_type */) {return false;},
            "fill_type"_a, supports_fill_type_doc)
        .def_static(
            "supports_line_type", [](contourpy::LineType /* line_type */) {return false;},
            "line_type"_a, supports_line_type_doc)
        .def_static("supports_quad_as_tri", []() {return false;}, supports_quad_as_tri_doc)
        .def_static("supports_threads", []() {return false;}, supports_threads_doc)
        .def_static("supports_z_interp", []() {return false;}, supports_z_interp_doc);

    py::class_<contourpy::Mpl2005ContourGenerator, contourpy::ContourGenerator>(
        m, "Mpl2005ContourGenerator",
        "ContourGenerator corresponding to ``name=\"mpl2005\"``.\n\n"
        "This is the original 2005 Matplotlib algorithm. "
        "Does not support any of ``corner_mask``, ``quad_as_tri``, ``threads`` or ``z_interp``. "
        "Only supports ``line_type=LineType.SeparateCode`` and ``fill_type=FillType.OuterCode``. "
        "Only supports chunking for filled contours, not contour lines.\n\n"
        ".. warning::\n"
        "   This algorithm is in ``contourpy`` for historic comparison. No new features or bug "
        "fixes will be added to it, except for security-related bug fixes.")
        .def(py::init<const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::MaskArray&,
                      contourpy::index_t,
                      contourpy::index_t>(),
             "x"_a, "y"_a, "z"_a, "mask"_a, py::kw_only(), "x_chunk_size"_a = 0,
             "y_chunk_size"_a = 0)
        .def_property_readonly(
            "chunk_count", &contourpy::Mpl2005ContourGenerator::get_chunk_count, chunk_count_doc)
        .def_property_readonly(
            "chunk_size", &contourpy::Mpl2005ContourGenerator::get_chunk_size, chunk_size_doc)
        .def_property_readonly(
            "fill_type", [](py::object /* self */) {return mpl20xx_fill_type;}, fill_type_doc)
        .def_property_readonly(
            "line_type", [](py::object /* self */) {return mpl20xx_line_type;}, line_type_doc)
        .def_property_readonly_static(
            "default_fill_type", [](py::object /* self */) {return mpl20xx_fill_type;},
            default_fill_type_doc)
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return mpl20xx_line_type;},
            default_line_type_doc)
        .def_static(
            "supports_fill_type",
            [](contourpy::FillType fill_type) {return fill_type == mpl20xx_fill_type;},
            supports_fill_type_doc)
        .def_static(
            "supports_line_type",
            [](contourpy::LineType line_type) {return line_type == mpl20xx_line_type;},
            supports_line_type_doc);

    py::class_<contourpy::mpl2014::Mpl2014ContourGenerator, contourpy::ContourGenerator>(
        m, "Mpl2014ContourGenerator",
        "ContourGenerator corresponding to ``name=\"mpl2014\"``.\n\n"
        "This is the 2014 Matplotlib algorithm, a replacement of the original 2005 algorithm that "
        "added ``corner_mask`` and made the code more maintainable. "
        "Only supports ``corner_mask``, does not support ``quad_as_tri``, ``threads`` or "
        "``z_interp``. \n"
        "Only supports ``line_type=LineType.SeparateCode`` and "
        "``fill_type=FillType.OuterCode``.\n\n"
        ".. warning::\n"
        "   This algorithm is in ``contourpy`` for historic comparison. No new features or bug "
        "fixes will be added to it, except for security-related bug fixes.")
        .def(py::init<const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::MaskArray&,
                      bool,
                      contourpy::index_t,
                      contourpy::index_t>(),
             "x"_a, "y"_a, "z"_a, "mask"_a, py::kw_only(), "corner_mask"_a, "x_chunk_size"_a = 0,
             "y_chunk_size"_a = 0)
        .def_property_readonly(
            "chunk_count", &contourpy::mpl2014::Mpl2014ContourGenerator::get_chunk_count,
            chunk_count_doc)
        .def_property_readonly(
            "chunk_size", &contourpy::mpl2014::Mpl2014ContourGenerator::get_chunk_size,
            chunk_size_doc)
        .def_property_readonly(
            "corner_mask", &contourpy::mpl2014::Mpl2014ContourGenerator::get_corner_mask,
            corner_mask_doc)
        .def_property_readonly(
            "fill_type", [](py::object /* self */) {return mpl20xx_fill_type;}, fill_type_doc)
        .def_property_readonly(
            "line_type", [](py::object /* self */) {return mpl20xx_line_type;}, line_type_doc)
        .def_property_readonly_static(
            "default_fill_type", [](py::object /* self */) {return mpl20xx_fill_type;},
            default_fill_type_doc)
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return mpl20xx_line_type;},
            default_line_type_doc)
        .def_static("supports_corner_mask", []() {return true;}, supports_corner_mask_doc)
        .def_static(
            "supports_fill_type",
            [](contourpy::FillType fill_type) {return fill_type == mpl20xx_fill_type;},
            supports_fill_type_doc)
        .def_static(
            "supports_line_type",
            [](contourpy::LineType line_type) {return line_type == mpl20xx_line_type;},
            supports_line_type_doc);

    py::class_<contourpy::SerialContourGenerator, contourpy::ContourGenerator>(
        m, "SerialContourGenerator",
        "ContourGenerator corresponding to ``name=\"serial\"``, the default algorithm for "
        "``contourpy``.\n\n"
        "Supports ``corner_mask``, ``quad_as_tri`` and ``z_interp`` but not ``threads``. "
        "Supports all options for ``line_type`` and ``fill_type``.")
        .def(py::init<const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::MaskArray&,
                      bool,
                      contourpy::LineType,
                      contourpy::FillType,
                      bool,
                      contourpy::ZInterp,
                      contourpy::index_t,
                      contourpy::index_t>(),
             "x"_a, "y"_a, "z"_a, "mask"_a, py::kw_only(), "corner_mask"_a, "line_type"_a,
             "fill_type"_a, "quad_as_tri"_a, "z_interp"_a, "x_chunk_size"_a = 0,
             "y_chunk_size"_a = 0)
        .def("_write_cache", &contourpy::SerialContourGenerator::write_cache)
        .def_property_readonly(
            "chunk_count", &contourpy::SerialContourGenerator::get_chunk_count, chunk_count_doc)
        .def_property_readonly(
            "chunk_size", &contourpy::SerialContourGenerator::get_chunk_size, chunk_size_doc)
        .def_property_readonly(
            "corner_mask", &contourpy::SerialContourGenerator::get_corner_mask, corner_mask_doc)
        .def_property_readonly(
            "fill_type", &contourpy::SerialContourGenerator::get_fill_type, fill_type_doc)
        .def_property_readonly(
            "line_type", &contourpy::SerialContourGenerator::get_line_type, line_type_doc)
        .def_property_readonly(
            "quad_as_tri", &contourpy::SerialContourGenerator::get_quad_as_tri, quad_as_tri_doc)
        .def_property_readonly(
            "z_interp", &contourpy::SerialContourGenerator::get_z_interp, z_interp_doc)
        .def_property_readonly_static(
            "default_fill_type",
            [](py::object /* self */) {return contourpy::SerialContourGenerator::default_fill_type();},
            default_fill_type_doc)
        .def_property_readonly_static("default_line_type",
            [](py::object /* self */) {return contourpy::SerialContourGenerator::default_line_type();},
            default_line_type_doc)
        .def_static("supports_corner_mask", []() {return true;}, supports_corner_mask_doc)
        .def_static(
            "supports_fill_type", &contourpy::SerialContourGenerator::supports_fill_type,
            supports_fill_type_doc)
        .def_static(
            "supports_line_type", &contourpy::SerialContourGenerator::supports_line_type,
            supports_line_type_doc)
        .def_static("supports_quad_as_tri", []() {return true;}, supports_quad_as_tri_doc)
        .def_static("supports_z_interp", []() {return true;}, supports_z_interp_doc);

    py::class_<contourpy::ThreadedContourGenerator, contourpy::ContourGenerator>(
        m, "ThreadedContourGenerator",
        "ContourGenerator corresponding to ``name=\"threaded\"``, the multithreaded version of "
        ":class:`~.SerialContourGenerator`.\n\n"
        "Supports ``corner_mask``, ``quad_as_tri`` and ``z_interp`` and ``threads``. "
        "Supports all options for ``line_type`` and ``fill_type``.")
        .def(py::init<const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::CoordinateArray&,
                      const contourpy::MaskArray&,
                      bool,
                      contourpy::LineType,
                      contourpy::FillType,
                      bool,
                      contourpy::ZInterp,
                      contourpy::index_t,
                      contourpy::index_t,
                      contourpy::index_t>(),
             "x"_a, "y"_a, "z"_a, "mask"_a, py::kw_only(), "corner_mask"_a, "line_type"_a,
             "fill_type"_a, "quad_as_tri"_a, "z_interp"_a, "x_chunk_size"_a = 0,
             "y_chunk_size"_a = 0, "thread_count"_a = 0)
        .def("_write_cache", &contourpy::ThreadedContourGenerator::write_cache)
        .def_property_readonly(
            "chunk_count", &contourpy::ThreadedContourGenerator::get_chunk_count, chunk_count_doc)
        .def_property_readonly(
            "chunk_size", &contourpy::ThreadedContourGenerator::get_chunk_size, chunk_size_doc)
        .def_property_readonly(
            "corner_mask", &contourpy::ThreadedContourGenerator::get_corner_mask, corner_mask_doc)
        .def_property_readonly(
            "fill_type", &contourpy::ThreadedContourGenerator::get_fill_type, fill_type_doc)
        .def_property_readonly(
            "line_type", &contourpy::ThreadedContourGenerator::get_line_type, line_type_doc)
        .def_property_readonly(
            "quad_as_tri", &contourpy::ThreadedContourGenerator::get_quad_as_tri, quad_as_tri_doc)
        .def_property_readonly(
            "thread_count", &contourpy::ThreadedContourGenerator::get_thread_count,
            thread_count_doc)
        .def_property_readonly(
            "z_interp", &contourpy::ThreadedContourGenerator::get_z_interp, z_interp_doc)
        .def_property_readonly_static(
            "default_fill_type",
            [](py::object /* self */) {return contourpy::ThreadedContourGenerator::default_fill_type();},
            default_fill_type_doc)
        .def_property_readonly_static(
            "default_line_type",
            [](py::object /* self */) {return contourpy::ThreadedContourGenerator::default_line_type();},
            default_line_type_doc)
        .def_static("supports_corner_mask", []() {return true;}, supports_corner_mask_doc)
        .def_static(
            "supports_fill_type", &contourpy::ThreadedContourGenerator::supports_fill_type,
            supports_fill_type_doc)
        .def_static(
            "supports_line_type", &contourpy::ThreadedContourGenerator::supports_line_type,
            supports_line_type_doc)
        .def_static("supports_quad_as_tri", []() {return true;}, supports_quad_as_tri_doc)
        .def_static("supports_threads", []() {return true;}, supports_threads_doc)
        .def_static("supports_z_interp", []() {return true;}, supports_z_interp_doc);
}
