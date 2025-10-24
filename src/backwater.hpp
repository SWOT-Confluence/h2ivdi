#ifndef BACKWATER_HPP
#define BACKWATER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array solve_backwater_rk22(py::array_t<double> x, py::array_t<double> H, py::array_t<double> W, py::array_t<double> S)


#endif