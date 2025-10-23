#ifndef STANDARD_STEP_HPP
#define STANDARD_STEP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> py_solve_standard_step_multi(py::array_t<double> py_x,
                                                 py::array_t<uint8_t> py_edge_type, 
                                                 py::array_t<double> py_He, 
                                                 py::array_t<double> py_We,
                                                 py::array_t<double> py_K,
                                                 py::array_t<double> py_d0,
                                                 py::array_t<double> py_Qin,
                                                 py::array_t<double> py_Hout);

py::array_t<double> py_solve_standard_step(py::array_t<double> py_x,
                                           py::array_t<uint8_t> py_edge_type, 
                                           py::array_t<double> py_He, 
                                           py::array_t<double> py_We,
                                           py::array_t<double> py_K,
                                           py::array_t<double> py_d0,
                                           double Qin,
                                           double Hout);

void solve_standard_step(size_t nx,
                         double* x,
                         uint8_t* edge_type,
                         double* He,
                         double* We,
                         double* d0,
                         double* K,
                         double Q,
                         double Hout,
                         double deps,
                         double eps,
                         uint16_t itermax,
                         double* H);


#endif