#include <pybind11/pybind11.h>

#include "common.hpp"
#include "standard_step.hpp"

namespace py = pybind11;

//using namespace h2ivdi;

PYBIND11_MODULE(H2iVDI_ext, m) {
//     m.doc() = R"pbdoc(
//         Pybind11 example plugin
//         -----------------------
//         .. currentmodule:: python_example
//         .. autosummary::
//            :toctree: _generate
//            add
//            subtract
//     )pbdoc";

    // py::class_<Options>(m, "Options")
    //     .def(py::init<>())
    //     .def_readwrite("verbose", &Options::verbose)
    //     .def_readwrite("allocationLimit", &Options::allocationLimit)
    //     .def("setLogInfo", &Options::setLogInfo);

    // m.def("defaultOptions", &eashydro::defaultOptions);
    // m.def("getCurrentAllocationMB", &eashydro::getCurrentAllocationMB);
    // // m.def("getCellsAreas", &get_cells_areas);
    // // m.def("getCellsPerimeters", &get_cells_perimeters);
    // // m.def("getCellsAspectRatios", &get_cells_aspect_ratio);
    // // m.def("getCellsMinimumAngles", &get_cells_minimum_angles);
    // // m.def("computeCellsIndicesFromCoordinates", &compute_cells_indices_from_coordinates);
    // // m.def("laplacianSmoothing", &laplacian_smoothing);
    m.def("solve_standard_step_multi", &py_solve_standard_step_multi);
    m.def("solve_standard_step", &py_solve_standard_step);
}