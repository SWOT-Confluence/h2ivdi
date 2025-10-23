#include <iostream>

#include "common.hpp"
#include "geometry.hpp"
#include "standard_step.hpp"


py::array_t<double> py_solve_standard_step_multi(py::array_t<double> py_x,
                                                 py::array_t<uint8_t> py_edge_type, 
                                                 py::array_t<double> py_He, 
                                                 py::array_t<double> py_We,
                                                 py::array_t<double> py_K,
                                                 py::array_t<double> py_d0,
                                                 py::array_t<double> py_Qin,
                                                 py::array_t<double> py_Hout) {


    // printf("!!! py_solve_standard_step_multi !!!\n");

    // Retrieve info and check x array
    py::buffer_info x_info = py_x.request();
    double* x = static_cast<double*>(x_info.ptr);
    std::vector<ssize_t> x_shape = x_info.shape;
    if (x_shape.size() != 1) throw std::runtime_error("'x' must be an array with ndim=1");
    // printf("x.shape=%i\n", x_shape[0]);

    // Retrieve info and check edge_type array
    py::buffer_info edge_type_info = py_edge_type.request();
    uint8_t* edge_type = static_cast<uint8_t*>(edge_type_info.ptr);
    std::vector<ssize_t> edge_type_shape = edge_type_info.shape;
    if (edge_type_shape.size() != 1) throw std::runtime_error("'edge_type_shape' must be an array with ndim=1");
    if (edge_type_shape[0] != x_shape[0]) throw std::runtime_error("size of 'edge_type' and size of 'x' must be agree");

    // Retrieve info and check He array
    py::buffer_info He_info = py_He.request();
    double* He = static_cast<double*>(He_info.ptr);
    std::vector<ssize_t> He_shape = He_info.shape;
    if (He_shape.size() != 2) throw std::runtime_error("'He' must be an array with ndim=2");
    if (He_shape[0] != 3) throw std::runtime_error("shape[0] of 'He' must be 3");
    if (He_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'He' and size of 'x' must be agree");
    // printf("He.shape=%i %i\n", He_shape[0], He_shape[1]);

    // Retrieve info and check We array
    py::buffer_info We_info = py_We.request();
    double* We = static_cast<double*>(We_info.ptr);
    std::vector<ssize_t> We_shape = We_info.shape;
    if (We_shape.size() != 2) throw std::runtime_error("'We' must be an array with ndim=2");
    if (We_shape[0] != 3) throw std::runtime_error("shape[0] of 'We' must be 3");
    if (We_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'We' and size of 'x' must be agree");

    // Retrieve info and check We array
    py::buffer_info K_info = py_K.request();
    double* K = static_cast<double*>(K_info.ptr);
    std::vector<ssize_t> K_shape = K_info.shape;
    if (K_shape.size() != 2) throw std::runtime_error("'K' must be an array with ndim=1");
    if (K_shape[0] != 2) throw std::runtime_error("shape[0] of 'K' must be 2");
    if (K_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'K' and size of 'x' must be agree");

    // Retrieve info and check We array
    py::buffer_info d0_info = py_d0.request();
    double* d0 = static_cast<double*>(d0_info.ptr);
    std::vector<ssize_t> d0_shape = d0_info.shape;
    if (d0_shape.size() != 1) throw std::runtime_error("'d0' must be an array with ndim=1");
    if (d0_shape[0] != x_shape[0]) throw std::runtime_error("shape[0] of 'd0' and size of 'x' must be agree");

    // Retrieve info and check Qin array
    py::buffer_info Qin_info = py_Qin.request();
    double* Qin = static_cast<double*>(Qin_info.ptr);
    std::vector<ssize_t> Qin_shape = Qin_info.shape;
    if (Qin_shape.size() != 1) throw std::runtime_error("'Qin' must be an array with ndim=1");

    // Retrieve info and check Hout array
    py::buffer_info Hout_info = py_Hout.request();
    double* Hout = static_cast<double*>(Hout_info.ptr);
    std::vector<ssize_t> Hout_shape = Hout_info.shape;
    if (Hout_shape.size() != 1) throw std::runtime_error("'Hout' must be an array with ndim=1");
    if (Hout_shape[0] != Qin_shape[0]) throw std::runtime_error("size of 'Hout' and size of 'Qin' must be agree");

    // for (size_t it = 0; it < Qin_shape[0]; it++) {
    //     printf("it=%i:Qin=%f,Hout=%f\n", it, Qin[it], Hout[it]);
    // }


    size_t nx = x_shape[0];
    size_t nt = Qin_shape[0];

    double* H = new double[Qin_shape[0] * x_shape[0]];

    for (size_t it = 0; it < nt; it++) {
        // printf("it=%i\n", it);
        solve_standard_step(nx, x, edge_type, He, We, d0, K, Qin[it], Hout[it], 0.01, 1e-3, 1000, &H[it*x_shape[0]]);
    }


    return py::array_t<double>(std::vector<ptrdiff_t>{Qin_shape[0], x_shape[0]}, H);

}


py::array_t<double> py_solve_standard_step(py::array_t<double> py_x,
                                           py::array_t<uint8_t> py_edge_type, 
                                           py::array_t<double> py_He, 
                                           py::array_t<double> py_We,
                                           py::array_t<double> py_K,
                                           py::array_t<double> py_d0,
                                           double Qin,
                                           double Hout) {

    // Retrieve info and check x array
    py::buffer_info x_info = py_x.request();
    double* x = static_cast<double*>(x_info.ptr);
    std::vector<ssize_t> x_shape = x_info.shape;
    if (x_shape.size() != 1) throw std::runtime_error("'x' must be an array with ndim=1");
    // printf("x.shape=%i\n", x_shape[0]);

    // Retrieve info and check edge_type array
    py::buffer_info edge_type_info = py_edge_type.request();
    uint8_t* edge_type = static_cast<uint8_t*>(edge_type_info.ptr);
    std::vector<ssize_t> edge_type_shape = edge_type_info.shape;
    if (edge_type_shape.size() != 1) throw std::runtime_error("'edge_type_shape' must be an array with ndim=1");
    if (edge_type_shape[0] != x_shape[0]) throw std::runtime_error("size of 'edge_type' and size of 'x' must be agree");

    // Retrieve info and check He array
    py::buffer_info He_info = py_He.request();
    double* He = static_cast<double*>(He_info.ptr);
    std::vector<ssize_t> He_shape = He_info.shape;
    if (He_shape.size() != 2) throw std::runtime_error("'He' must be an array with ndim=2");
    if (He_shape[0] != 3) throw std::runtime_error("shape[0] of 'He' must be 3");
    if (He_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'He' and size of 'x' must be agree");
    // printf("He.shape=%i %i\n", He_shape[0], He_shape[1]);

    // Retrieve info and check We array
    py::buffer_info We_info = py_We.request();
    double* We = static_cast<double*>(We_info.ptr);
    std::vector<ssize_t> We_shape = We_info.shape;
    if (We_shape.size() != 2) throw std::runtime_error("'We' must be an array with ndim=2");
    if (We_shape[0] != 3) throw std::runtime_error("shape[0] of 'We' must be 3");
    if (We_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'We' and size of 'x' must be agree");

    // Retrieve info and check We array
    py::buffer_info K_info = py_K.request();
    double* K = static_cast<double*>(K_info.ptr);
    std::vector<ssize_t> K_shape = K_info.shape;
    if (K_shape.size() != 2) throw std::runtime_error("'K' must be an array with ndim=1");
    if (K_shape[0] != 2) throw std::runtime_error("shape[0] of 'K' must be 2");
    if (K_shape[1] != x_shape[0]) throw std::runtime_error("shape[1] of 'K' and size of 'x' must be agree");

    // Retrieve info and check We array
    py::buffer_info d0_info = py_d0.request();
    double* d0 = static_cast<double*>(d0_info.ptr);
    std::vector<ssize_t> d0_shape = d0_info.shape;
    if (d0_shape.size() != 1) throw std::runtime_error("'d0' must be an array with ndim=1");
    if (d0_shape[0] != x_shape[0]) throw std::runtime_error("shape[1] of 'd0' and size of 'x' must be agree");

    size_t nx = x_shape[0];

    double* H = new double[x_shape[0]];

    solve_standard_step(nx, x, edge_type, He, We, d0, K, Qin, Hout, 0.01, 1e-3, 1000, H);

    return py::array_t<double>(std::vector<ptrdiff_t>{x_shape[0]}, H);

}

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
                         double* H) {

    size_t ie;
    uint16_t iter;
    double A1L, A1R, A2L, A2R;
    double bL, bR;
    double dc;
    double dL, dLm1, dLm2, dR;
    double dx;
    double Deb, DebL, DebR;
    double errm1, errm2;
    double HL, HR;
    double P1L, P1R, P2L, P2R;
    double Rh1L, Rh1R, Rh2L, Rh2R;
    double Sf;
    double uL, uR;

    // Downstream BC
    ie = nx - 1;
    H[ie] = Hout;
    HR = Hout;
    bR = He[ie] - d0[ie];
    dR = Hout - bR;
    // Compute compound flow areas, wetted perimeters and widths at right state
    compound_area31(HR, bR, He[ie], He[nx+ie], He[2*nx+ie], We[ie], We[nx+ie], We[2*nx+ie], &A1R, &A2R);
    compound_perim31(HR, bR, He[ie], He[nx+ie], He[2*nx+ie], We[ie], We[nx+ie], We[2*nx+ie], &P1R, &P2R);
    // compound_width31(HR, He[ie], He[nx+ie], He[2*nx+ie], We[ie], We[nx+ie], We[2*nx+ie], &WR1, &WR2);

    // Compute compound hydraulic radiuses at right state
    Rh1R = (P1R > 1e-12) ? A1R / P1R : 0.0;
    Rh2R = (P2R > 1e-12) ? A2R / P2R : 0.0;
    // AR = A1R + A2R;
    // PR = A1R + A2R;

    // Compute total debitance at right state
    DebR = K[ie] * A1R * pow(Rh1R, CONSTANT_2D3) + K[nx+ie] * A2R * pow(Rh2R, CONSTANT_2D3);

    // Compute velocity at right state
    uR = (A1R > 1e-12) ? Q / A1R : 0.0;
#ifdef DEBUG
    printf("OUT:H=%f,d=%f,b=%f,h0=%f,u=%f,K=(%.1f,%.1f)A=(%f,%f),P=(%f,%f)\n", HR, dR, bR, d0[ie], uR, K[ie], K[nx+ie], A1R, A2R, P1R, P2R);
#endif

    for (ie = nx-1; ie > 0; ie--) {
#ifdef DEBUG
        printf("ie=%i\n", ie);
#endif
        dLm1 = dR;
        dLm2 = dR + 0.1;
        dL = dR;
        bL = He[ie-1] - d0[ie-1];
        dx = fabs(x[ie-1] - x[ie]);
        iter = 0;
        errm1 = errm2 = 0.0;
        while (fabs(dLm1 - dLm2) > eps && iter < itermax) {
            iter++;
            HL = dLm1 + bL;

            // Compute compound flow areas, wetted perimeters and widths at left state
            compound_area31(HL, bL, He[ie-1], He[nx+ie-1], He[2*nx+ie-1], We[ie-1], We[nx+ie-1], We[2*nx+ie-1], &A1L, &A2L);
            compound_perim31(HL, bL, He[ie-1], He[nx+ie-1], He[2*nx+ie-1], We[ie-1], We[nx+ie-1], We[2*nx+ie-1], &P1L, &P2L);
            // compound_width31(HR, He[ie], He[nx+ie], He[2*nx+ie], We[ie], We[nx+ie], We[2*nx+ie], &WR1, &WR2);

            // Compute compound hydraulic radiuses at left state
            Rh1L = (P1L > 1e-12) ? A1L / P1L : 0.0;
            Rh2L = (P2L > 1e-12) ? A2L / P2L : 0.0;
            // AR = A1R + A2R;
            // PR = A1R + A2R;
            // Compute total debitance at left state
            // printf("ie, K1=%f (A1=%f), K2=%f (A2=%f)\n", K[ie-1], A1L, K[nx+ie-1], A2L);
            DebL = K[ie-1] * A1L * pow(Rh1L, CONSTANT_2D3) + K[nx+ie-1] * A2L * pow(Rh2L, CONSTANT_2D3);

            // Compute velocity at left state
            uL = (A1L > 1e-12) ? Q / A1L : 0.0;

            // Compute average debitance and friction head
            Deb = 0.5 * (DebL + DebR);
            Sf = pow(Q / Deb, 2);

            // Compute height at left state
            HL = HR + 0.5 * (uR*uR - uL*uL) / CONSTANT_GRAVITY + dx * Sf;
            dL = HL - bL;
            critical_depth31(Q, bL, He[ie-1], He[nx+ie-1], He[2*nx+ie-1], We[ie-1], We[nx+ie-1], We[2*nx+ie-1], &dc);
            if (dL < dc) dL = dc;
            if (dL < deps) dL = deps;

#ifdef DEBUG
            printf("iter=%i\n", iter);
            printf("dx=%f\n", dx);
            printf("L:d=%f,u=%f,A=(%f,%f),P=(%f,%f)\n", dL, uL, A1L, A2L, P1L, P2L);
            printf("R:d=%f,u=%f,A=(%f,%f),P=(%f,%f)\n", dR, uR, A1R, A2R, P1R, P2R);

            printf(" ** dZ %f\n", 0.5 * (uR*uR - uL*uL) / CONSTANT_GRAVITY + dx * Sf);
            printf("--DETAILS %f %f %21.16e %f\n", uR, uL, Sf, dx);
            printf(" ** DebL %f %f %f\n", DebL, K[ie-1] * A1L * pow(Rh1L, CONSTANT_2D3), K[nx+ie-1] * A2L * pow(Rh2L, CONSTANT_2D3));
            printf(" ** DETAIL_L %f %f %f\n", K[ie-1], A1L, Rh1L);
            printf(" ** DebR %f %f %f\n", DebR, K[ie] * A1R * pow(Rh1R, CONSTANT_2D3), K[nx+ie] * A2R * pow(Rh2R, CONSTANT_2D3));
            printf(" ** DETAIL_R %f %f %f\n", K[ie], A1R, Rh1R);
            printf(" ** z %f\n", HL);
            printf(" ** h,hc %f %f\n", dL, dc);
            do 
            {
                std::cout << '\n' << "Press a key to continue...";
            } while (std::cin.get() != '\n');
#endif



            // TODO: compute critical depth (dc) and enforce dL > dc
            
            // Update depth at left state
            if (iter == 1) {
                errm1 = dL - dLm1;
                dLm1 += 0.7 * errm1;
                if (dLm1 < deps) dLm1 = deps;
            } else {
                errm2 = errm1;
                errm1 = dL - dLm1;
                if (fabs(errm2 - errm1) < 1e-2) {
                    dL = dLm2 + 0.5 * errm2;
                } else {
                    dL = dLm2 - errm2 * (dLm2 - dLm1) / (errm2 - errm1);
                }

                // TODO: compute critical depth (dc) and enforce dL > dc
                critical_depth31(Q, bL, He[ie-1], He[nx+ie-1], He[2*nx+ie-1], We[ie-1], We[nx+ie-1], We[2*nx+ie-1], &dc);
                if (dL < dc) dL = dc;
                if (dL < deps) dL = deps;

                // Update dLm1 and dLm2 for next step
                dLm2 = dLm1;
                dLm1 = dL;
            }

        }

        // Store height at left state
        H[ie-1] = dL + bL;
#ifdef DEBUG
        printf("IN[%i]:H=%f,d=%f,b=%f\n", ie-1, H[ie-1], dL, bL);
#endif

        // Prepare next right state
        dR = dL;
        HR = HL;
        DebR = DebL;
        uR = uL;

    }

}
