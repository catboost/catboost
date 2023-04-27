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

static contourpy::LineType mpl20xx_line_type = contourpy::LineType::SeparateCode;
static contourpy::FillType mpl20xx_fill_type = contourpy::FillType::OuterCode;

PYBIND11_MODULE(_contourpy, m) {
    m.doc() =
        "C++11 extension module wrapped using `pybind11`_.\n\n"
        ".. note::\n"
        "   It should not be necessary to access classes and functions in this extension module "
        "directly. Instead, :func:`contourpy.contour_generator` should be used to create "
        ":class:`~contourpy.ContourGenerator` objects, and the enums "
        "(:class:`~contourpy.FillType`, :class:`~contourpy.LineType` and "
        ":class:`~contourpy.ZInterp`) and :func:`contourpy.max_threads` function are all available "
        "in the :mod:`contourpy` module.";

    m.attr("CONTOURPY_DEBUG") = CONTOURPY_DEBUG;
    m.attr("CONTOURPY_CXX11") = CONTOURPY_CXX11;
    m.attr("__version__") = MACRO_STRINGIFY(CONTOURPY_VERSION);

    py::enum_<contourpy::FillType>(m, "FillType",
        "Enum used for ``fill_type`` keyword argument in :func:`~contourpy.contour_generator`.\n\n"
        "This controls the format of filled contour data returned from "
        ":meth:`~contourpy.ContourGenerator.filled`.")
        .value("OuterCode", contourpy::FillType::OuterCode)
        .value("OuterOffset", contourpy::FillType::OuterOffset)
        .value("ChunkCombinedCode", contourpy::FillType::ChunkCombinedCode)
        .value("ChunkCombinedOffset", contourpy::FillType::ChunkCombinedOffset)
        .value("ChunkCombinedCodeOffset", contourpy::FillType::ChunkCombinedCodeOffset)
        .value("ChunkCombinedOffsetOffset", contourpy::FillType::ChunkCombinedOffsetOffset)
        .export_values();

    py::enum_<contourpy::LineType>(m, "LineType",
        "Enum used for ``line_type`` keyword argument in :func:`~contourpy.contour_generator`.\n\n"
        "This controls the format of contour line data returned from "
        ":meth:`~contourpy.ContourGenerator.lines`.")
        .value("Separate", contourpy::LineType::Separate)
        .value("SeparateCode", contourpy::LineType::SeparateCode)
        .value("ChunkCombinedCode", contourpy::LineType::ChunkCombinedCode)
        .value("ChunkCombinedOffset", contourpy::LineType::ChunkCombinedOffset)
        .export_values();

    py::enum_<contourpy::ZInterp>(m, "ZInterp",
        "Enum used for ``z_interp`` keyword argument in :func:`~contourpy.contour_generator`\n\n"
        "This controls the interpolation used on ``z`` values to determine where contour lines "
        "intersect the edges of grid quads, and ``z`` values at quad centres.")
        .value("Linear", contourpy::ZInterp::Linear)
        .value("Log", contourpy::ZInterp::Log)
        .export_values();

    m.def("max_threads", &contourpy::Util::get_max_threads,
        "Return the maximum number of threads, obtained from "
        "``std::thread::hardware_concurrency()``.\n\n"
        "This is the number of threads used by a multithreaded ContourGenerator if the kwarg "
        "``threads=0`` is passed to :func:`~contourpy.contour_generator`.");

    py::class_<contourpy::ContourGenerator>(m, "ContourGenerator",
        "Abstract base class for contour generator classes, defining the interface that they all "
        "implement.")
        .def("create_contour", [](double level) {return py::make_tuple();},
            "Synonym for :func:`~contourpy.ContourGenerator.lines` to provide backward "
            "compatibility with Matplotlib.")
        .def("create_filled_contour",
            [](double lower_level, double upper_level) {return py::make_tuple();},
            "Synonym for :func:`~contourpy.ContourGenerator.filled` to provide backward "
            "compatibility with Matplotlib.")
        .def("filled", [](double lower_level, double upper_level) {return py::make_tuple();},
            "Calculate and return filled contours between two levels.\n\n"
            "Args:\n"
            "    lower_level (float): Lower z-level of the filled contours.\n"
            "    upper_level (float): Upper z-level of the filled contours.\n"
            "Return:\n"
            "    Filled contour polygons as one or more sequences of numpy arrays. The exact "
            "format is determined by the ``fill_type`` used by the ``ContourGenerator``.")
        .def("lines", [](double level) {return py::make_tuple();},
            "Calculate and return contour lines at a particular level.\n\n"
            "Args:\n"
            "    level (float): z-level to calculate contours at.\n\n"
            "Return:\n"
            "    Contour lines (open line strips and closed line loops) as one or more sequences "
            "of numpy arrays. The exact format is determined by the ``line_type`` used by the "
            "``ContourGenerator``.")
        .def_property_readonly(
            "chunk_count", [](py::object /* self */) {return py::make_tuple(1, 1);},
            "Return tuple of (y, x) chunk counts.")
        .def_property_readonly(
            "chunk_size", [](py::object /* self */) {return py::make_tuple(1, 1);},
            "Return tuple of (y, x) chunk sizes.")
        .def_property_readonly(
            "corner_mask", [](py::object /* self */) {return false;},
            "Return whether ``corner_mask`` is set or not.")
        .def_property_readonly(
            "fill_type", [](py::object /* self */) {return contourpy::FillType::OuterOffset;},
            "Return the ``FillType``.")
        .def_property_readonly(
            "line_type", [](py::object /* self */) {return contourpy::LineType::Separate;},
            "Return the ``LineType``.")
        .def_property_readonly(
            "quad_as_tri", [](py::object /* self */) {return false;},
            "Return whether ``quad_as_tri`` is set or not.")
        .def_property_readonly(
            "thread_count", [](py::object /* self */) {return 1;},
            "Return the number of threads used.")
        .def_property_readonly(
            "z_interp", [](py::object /* self */) {return contourpy::ZInterp::Linear;},
            "Return the ``ZInterp``.")
        .def_property_readonly_static(
            "default_fill_type",
            [](py::object /* self */) {return contourpy::FillType::OuterOffset;},
            "Return the default ``FillType`` used by this algorithm.")
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return contourpy::LineType::Separate;},
            "Return the default ``LineType`` used by this algorithm.")
        .def_static(
            "supports_corner_mask", []() {return false;},
            "Return whether this algorithm supports ``corner_mask``.")
        .def_static(
            "supports_fill_type", [](contourpy::FillType /* fill_type */) {return false;},
            "Return whether this algorithm supports a particular ``FillType``.")
        .def_static(
            "supports_line_type", [](contourpy::LineType /* line_type */) {return false;},
            "Return whether this algorithm supports a particular ``LineType``.")
        .def_static(
            "supports_quad_as_tri", []() {return false;},
            "Return whether this algorithm supports ``quad_as_tri``.")
        .def_static(
            "supports_threads", []() {return false;},
            "Return whether this algorithm supports the use of threads.")
        .def_static(
            "supports_z_interp", []() {return false;},
            "Return whether this algorithm supports ``z_interp`` values other than "
            "``ZInterp.Linear`` which all support.");

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
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             py::arg("mask"),
             py::kw_only(),
             py::arg("x_chunk_size") = 0,
             py::arg("y_chunk_size") = 0)
        .def("create_contour", &contourpy::Mpl2005ContourGenerator::lines,
            "Synonym for :func:`~contourpy.Mpl2005ContourGenerator.lines` to provide backward "
            "compatibility with Matplotlib.")
        .def("create_filled_contour", &contourpy::Mpl2005ContourGenerator::filled,
            "Synonym for :func:`~contourpy.Mpl2005ContourGenerator.filled` to provide backward "
            "compatibility with Matplotlib.")
        .def("filled", &contourpy::Mpl2005ContourGenerator::filled)
        .def("lines", &contourpy::Mpl2005ContourGenerator::lines)
        .def_property_readonly("chunk_count", &contourpy::Mpl2005ContourGenerator::get_chunk_count)
        .def_property_readonly("chunk_size", &contourpy::Mpl2005ContourGenerator::get_chunk_size)
        .def_property_readonly("fill_type", [](py::object /* self */) {return mpl20xx_fill_type;})
        .def_property_readonly("line_type", [](py::object /* self */) {return mpl20xx_line_type;})
        .def_property_readonly_static(
            "default_fill_type", [](py::object /* self */) {return mpl20xx_fill_type;})
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return mpl20xx_line_type;})
        .def_static(
            "supports_fill_type",
            [](contourpy::FillType fill_type) {return fill_type == mpl20xx_fill_type;})
        .def_static(
            "supports_line_type",
            [](contourpy::LineType line_type) {return line_type == mpl20xx_line_type;});

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
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             py::arg("mask"),
             py::kw_only(),
             py::arg("corner_mask"),
             py::arg("x_chunk_size") = 0,
             py::arg("y_chunk_size") = 0)
        .def("create_contour", &contourpy::mpl2014::Mpl2014ContourGenerator::lines,
            "Synonym for :func:`~contourpy.Mpl2014ContourGenerator.lines` to provide backward "
            "compatibility with Matplotlib.")
        .def("create_filled_contour", &contourpy::mpl2014::Mpl2014ContourGenerator::filled,
            "Synonym for :func:`~contourpy.Mpl2014ContourGenerator.filled` to provide backward "
            "compatibility with Matplotlib.")
        .def("filled", &contourpy::mpl2014::Mpl2014ContourGenerator::filled)
        .def("lines", &contourpy::mpl2014::Mpl2014ContourGenerator::lines)
        .def_property_readonly(
            "chunk_count", &contourpy::mpl2014::Mpl2014ContourGenerator::get_chunk_count)
        .def_property_readonly(
            "chunk_size", &contourpy::mpl2014::Mpl2014ContourGenerator::get_chunk_size)
        .def_property_readonly(
            "corner_mask", &contourpy::mpl2014::Mpl2014ContourGenerator::get_corner_mask)
        .def_property_readonly("fill_type", [](py::object /* self */) {return mpl20xx_fill_type;})
        .def_property_readonly("line_type", [](py::object /* self */) {return mpl20xx_line_type;})
        .def_property_readonly_static(
            "default_fill_type", [](py::object /* self */) {return mpl20xx_fill_type;})
        .def_property_readonly_static(
            "default_line_type", [](py::object /* self */) {return mpl20xx_line_type;})
        .def_static("supports_corner_mask", []() {return true;})
        .def_static(
            "supports_fill_type",
            [](contourpy::FillType fill_type) {return fill_type == mpl20xx_fill_type;})
        .def_static(
            "supports_line_type",
            [](contourpy::LineType line_type) {return line_type == mpl20xx_line_type;});

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
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             py::arg("mask"),
             py::kw_only(),
             py::arg("corner_mask"),
             py::arg("line_type"),
             py::arg("fill_type"),
             py::arg("quad_as_tri"),
             py::arg("z_interp"),
             py::arg("x_chunk_size") = 0,
             py::arg("y_chunk_size") = 0)
        .def("_write_cache", &contourpy::SerialContourGenerator::write_cache)
        .def("create_contour", &contourpy::SerialContourGenerator::lines)
        .def("create_filled_contour", &contourpy::SerialContourGenerator::filled)
        .def("filled", &contourpy::SerialContourGenerator::filled)
        .def("lines", &contourpy::SerialContourGenerator::lines)
        .def_property_readonly("chunk_count", &contourpy::SerialContourGenerator::get_chunk_count)
        .def_property_readonly("chunk_size", &contourpy::SerialContourGenerator::get_chunk_size)
        .def_property_readonly("corner_mask", &contourpy::SerialContourGenerator::get_corner_mask)
        .def_property_readonly("fill_type", &contourpy::SerialContourGenerator::get_fill_type)
        .def_property_readonly("line_type", &contourpy::SerialContourGenerator::get_line_type)
        .def_property_readonly("quad_as_tri", &contourpy::SerialContourGenerator::get_quad_as_tri)
        .def_property_readonly("z_interp", &contourpy::SerialContourGenerator::get_z_interp)
        .def_property_readonly_static("default_fill_type", [](py::object /* self */) {
            return contourpy::SerialContourGenerator::default_fill_type();})
        .def_property_readonly_static("default_line_type", [](py::object /* self */) {
            return contourpy::SerialContourGenerator::default_line_type();})
        .def_static("supports_corner_mask", []() {return true;})
        .def_static("supports_fill_type", &contourpy::SerialContourGenerator::supports_fill_type)
        .def_static("supports_line_type", &contourpy::SerialContourGenerator::supports_line_type)
        .def_static("supports_quad_as_tri", []() {return true;})
        .def_static("supports_z_interp", []() {return true;});

    py::class_<contourpy::ThreadedContourGenerator, contourpy::ContourGenerator>(
        m, "ThreadedContourGenerator",
        "ContourGenerator corresponding to ``name=\"threaded\"``, the multithreaded version of "
        ":class:`~contourpy._contourpy.SerialContourGenerator`.\n\n"
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
             py::arg("x"),
             py::arg("y"),
             py::arg("z"),
             py::arg("mask"),
             py::kw_only(),
             py::arg("corner_mask"),
             py::arg("line_type"),
             py::arg("fill_type"),
             py::arg("quad_as_tri"),
             py::arg("z_interp"),
             py::arg("x_chunk_size") = 0,
             py::arg("y_chunk_size") = 0,
             py::arg("thread_count") = 0)
        .def("_write_cache", &contourpy::ThreadedContourGenerator::write_cache)
        .def("create_contour", &contourpy::ThreadedContourGenerator::lines,
            "Synonym for :func:`~contourpy.ThreadedContourGenerator.lines` to provide backward "
            "compatibility with Matplotlib.")
        .def("create_filled_contour", &contourpy::ThreadedContourGenerator::filled,
            "Synonym for :func:`~contourpy.ThreadedContourGenerator.filled` to provide backward "
            "compatibility with Matplotlib.")
        .def("filled", &contourpy::ThreadedContourGenerator::filled)
        .def("lines", &contourpy::ThreadedContourGenerator::lines)
        .def_property_readonly("chunk_count", &contourpy::ThreadedContourGenerator::get_chunk_count)
        .def_property_readonly("chunk_size", &contourpy::ThreadedContourGenerator::get_chunk_size)
        .def_property_readonly("corner_mask", &contourpy::ThreadedContourGenerator::get_corner_mask)
        .def_property_readonly("fill_type", &contourpy::ThreadedContourGenerator::get_fill_type)
        .def_property_readonly("line_type", &contourpy::ThreadedContourGenerator::get_line_type)
        .def_property_readonly("quad_as_tri", &contourpy::ThreadedContourGenerator::get_quad_as_tri)
        .def_property_readonly(
            "thread_count", &contourpy::ThreadedContourGenerator::get_thread_count)
        .def_property_readonly("z_interp", &contourpy::ThreadedContourGenerator::get_z_interp)
        .def_property_readonly_static("default_fill_type", [](py::object /* self */) {
            return contourpy::ThreadedContourGenerator::default_fill_type();})
        .def_property_readonly_static("default_line_type", [](py::object /* self */) {
            return contourpy::ThreadedContourGenerator::default_line_type();})
        .def_static("supports_corner_mask", []() {return true;})
        .def_static("supports_fill_type", &contourpy::ThreadedContourGenerator::supports_fill_type)
        .def_static("supports_line_type", &contourpy::ThreadedContourGenerator::supports_line_type)
        .def_static("supports_quad_as_tri", []() {return true;})
        .def_static("supports_threads", []() {return true;})
        .def_static("supports_z_interp", []() {return true;});
}
