#include "backwater.hpp"

py::array solve_backwater_rk4(py::array_t<double> py_x,
                              py::array_t<double> py_h,
                              py::array_t<double> py_k,
                              py::array_t<double> py_H0,
                              py::array_t<double> py_Wr) {

    // Retrieve info and check x array
    py::buffer_info x_info = py_x.request();
    uint8_t* x = static_cast<double*>(x_info.ptr);
    std::vector<ssize_t> x_shape = x_info.shape;
    if (x_shape.size() != 1) throw std::runtime_error("'x' must be an array with ndim=1");

    // Retrieve info and check h array
    py::buffer_info h_info = py_h.request();
    uint8_t* h = static_cast<double*>(h_info.ptr);
    std::vector<ssize_t> h_shape = h_info.shape;
    if (h_shape.size() != 1) throw std::runtime_error("'h' must be an array with ndim=1");
    if (h_shape[0] != x_shape[0]) throw std::runtime_error("'h' and 'x' must be of same size");

    // Retrieve info and check k array
    py::buffer_info k_info = py_k.request();
    uint8_t* k = static_cast<double*>(k_info.ptr);
    std::vector<ssize_t> k_shape = k_info.shape;
    if (k_shape.size() != 1) throw std::runtime_error("'k' must be an array with ndim=1");
    if (k_shape[0] != x_shape[0]) throw std::runtime_error("'k' and 'x' must be of same size");

    // Retrieve info and check H0 array
    py::buffer_info H0_info = py_H0.request();
    uint8_t* H0 = static_cast<double*>(H0_info.ptr);
    std::vector<ssize_t> H0_shape = H0_info.shape;
    if (H0_shape.size() != 1) throw std::runtime_error("'H0' must be an array with ndim=1");
    if (H0_shape[0] != x_shape[0]) throw std::runtime_error("'H0' and 'x' must be of same size");

    // Retrieve info and check Wr array
    py::buffer_info Wr_info = py_Wr.request();
    uint8_t* Wr = static_cast<double*>(Wr_info.ptr);
    std::vector<ssize_t> Wr_shape = Wr_info.shape;
    if (Wr_shape.size() != 1) throw std::runtime_error("'Wr' must be an array with ndim=1");
    if (Wr_shape[0] != x_shape[0]) throw std::runtime_error("'Wr' and 'x' must be of same size");

    size_t nt = H_shape[0];
    size_t nx = H_shape[1];

    double Hest = new double[H_shape[0] * H_shape[1]];
    double b1, b2;
    double k1, k2;
    double w1, w2;
    double x1, x2;
    double y1, y2;

    for (size_t it = 0; it < nt; it++) {
        Hest[it * nx + nx - 1] = H[it * nx + nx - 1];
        for (size_t ix = 0; ix < nx; ix++) {
            x1 = x[r]
            x2 = x[r-1]
            y1 = H[it * nx + r]
            b2 = H0r[r-1] - h[r-1]
            b1 = H0r[r] - h[r]
            W2 = Wr[r-1]
            W1 = Wr[r]
            k2 = k[r-1]
            k1 = k[r]
            if method == "rk22":
                H[it, r-1] = self.solveRK22(y1, x1, x2, b1, b2, W1, W2, K1, K2, Q[it])
            else:
                H[it, r-1] = self.solveRK11(y1, x1, x2, b1, b2, W1, W2, K1, K2, Q[it])    
        }

    }
            for r in range(self._data.H.shape[1]-1, 0, -1):
                x1 = self._data.x[r]
                x2 = self._data.x[r-1]
                y1 = H[it, r]
                b2 = self._data.H0r[r-1] - self._h0 * self._h1[r-1]
                b1 = self._data.H0r[r] - self._h0 * self._h1[r]
                W2 = self._data.Wr[r-1]
                W1 = self._data.Wr[r]
                K2 = self._k0 * self._k1[r-1]
                K1 = self._k0 * self._k1[r]
                if method == "rk22":
                    H[it, r-1] = self.solveRK22(y1, x1, x2, b1, b2, W1, W2, K1, K2, Q[it])
                else:
                    H[it, r-1] = self.solveRK11(y1, x1, x2, b1, b2, W1, W2, K1, K2, Q[it])    

}