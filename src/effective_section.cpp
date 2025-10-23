#include <cmath>
#include <cstdint>

#include "common.hpp"
#include "effective_section.hpp"


py::array_t<double> py_effective_dry_section3(py::array_t<double> py_Hs,
                                              py::array_t<double> py_Ws) {


    // Retrieve info and check Hs array
    py::buffer_info Hs_info = py_Hs.request();
    double* Hs = static_cast<double*>(Hs_info.ptr);
    std::vector<ssize_t> Hs_shape = Hs_info.shape;
    if (Hs_shape.size() != 1) throw std::runtime_error("'Hs' must be an array with ndim=1");

    // Retrieve info and check Ws array
    py::buffer_info Ws_info = py_Ws.request();
    double* Ws = static_cast<double*>(Ws_info.ptr);
    std::vector<ssize_t> Ws_shape = Ws_info.shape;
    if (Ws_shape.size() != 1) throw std::runtime_error("'Ws' must be an array with ndim=1");
    if (Ws_shape[0] != Hs_shape[0]) throw std::runtime_error("size of 'Ws' and size of 'Hs' must be agree");

    double* HWe = new double[6];
    double* He = HWe;
    double* We = HWe+3*sizeof(double);

    effective_dry_section3(Hs_shape[0], Hs, Ws, He, We);

    return py::array_t<double>(std::vector<ptrdiff_t>{Hs_shape[0], 3}, HWe);

}

void effective_dry_section3(ssize_t n, double* Hs, double* Ws, double* He, double* We) {
    double Hmean, Wmean;
    double alpha, beta;

    for (size_t i = 1; i < n-1; i++) {
        Hm = Hs[i];

        // Compute average H and W for lowest part
        Hmean = 0.0;
        Wmean = 0.0;
        for (size_t j = 0; j <= i; j++) {
            Hmean += Hs[j];
            Wmean += Ws[j];
        }
        Hmean /= (double)
    }
}
