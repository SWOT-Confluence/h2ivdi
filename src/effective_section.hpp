#ifndef EFFECTIVE_SECTION_HPP
#define EFFECTIVE_SECTION_HPP


py::array_t<double> py_effective_dry_section3(py::array_t<double> py_Hs,
                                              py::array_t<double> py_Ws);

void effective_dry_section3(double Hs, double Ws, double * He, double * We);

#endif