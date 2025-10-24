#include <cmath>
#include <cstdint>

#include "common.hpp"
#include "geometry.hpp"


void area31(double H,
            double b,
            double He0,
            double He1,
            double He2,
            double We0,
            double We1,
            double We2,
            double* A) {

    double dH;
    double W;
    if (H < He0) {
        *A = (H - b) * We0;
    } else if (H < He1) {
        dH = H - He0;
        W = We0 + dH / (He1 - He0) * (We1 - We0);
        *A = (He0 - b) * We0 + dH * 0.5 * (W + We0);
    } else if (H < He2) {
        dH = H - He1;
        W = We1 + dH / (He2 - He1) * (We2 - We1);
        *A = (He0 - b) * We0 + 
             (He1 - He0) * 0.5 * (We1 + We0) + 
             dH * 0.5 * (W + We1);
    } else {
        dH = H - He2;
        W = We2 + dH / (He2 - He1) * (We2 - We1);
        *A = (He0 - b) * We0 + 
             (He1 - He0) * 0.5 * (We1 + We0) + 
             (He2 - He1) * 0.5 * (We2 + We1) +
             dH * 0.5 * (W + We2);
    }
}

void width31(double H,
             double b,
             double He0,
             double He1,
             double He2,
             double We0,
             double We1,
             double We2,
             double* W) {

    double dH;
    if (H < He0) {
        *W = We0;
    } else if (H < He1) {
        dH = H - He0;
        *W = We0 + dH / (He1 - He0) * (We1 - We0);
    } else if (H < He2) {
        dH = H - He1;
        *W = We1 + dH / (He2 - He1) * (We2 - We1);
    } else {
        dH = H - He2;
        *W = We2 + dH / (He2 - He1) * (We2 - We1);
    }
}

void compound_area31(double H,
                     double b,
                     double He0,
                     double He1,
                     double He2,
                     double We0,
                     double We1,
                     double We2,
                     double* A1,
                     double* A2) {

    double dH;
    double W;
    if (H < He0) {
        *A1 = (H - b) * We0;
        *A2 = 0.0;
    } else if (H < He1) {
        dH = H - He0;
        W = We0 + dH / (He1 - He0) * (We1 - We0);
        *A1 = (He0 - b) * We0 + dH * 0.5 * (W + We0);
        *A2 = 0.0;
    } else if (H < He2) {
        dH = H - He1;
        W = We1 + dH / (He2 - He1) * (We2 - We1);
        *A1 = (He0 - b) * We0 + (He1 - He0) * 0.5 * (We0 + We1) + dH * We1;
        // A2 = dH * 0.5 * (W + We1) - dH * We1 = dH * 0.5 * (W - We1)
        *A2 = dH * 0.5 * (W - We1);
    } else {
        dH = H - He2;
        W = We2 + dH / (He2 - He1) * (We2 - We1);
        *A1 = (He0 - b) * We0 + (He1 - He0) * 0.5 * (We0 + We1) + (dH + He2 - He1) * We1;
        // A2 = (He2 - He1) * 0.5 * (We2 + We1) + dH * 0.5 * (W + We2) - (He2 - He1) * We1 - dH * We1
        // => A2 = (He2 - He1) * 0.5 * (We2 - We1) + dH * (0.5 * (W + We2) - We1)
        *A2 = (He2 - He1) * 0.5 * (We2 - We1) + dH * (0.5 * (W + We2) - We1);
    }
}

void compound_perim31(double H,
                      double b,
                      double He0,
                      double He1,
                      double He2,
                      double We0,
                      double We1,
                      double We2,
                      double* P1,
                      double* P2) {
    double dH;
    double dW;
    double W;
    if (H < He0) {
        *P1 = 2.0 * (H - b) + We0;
        *P2 = 0.0;
    } else if (H < He1) {
        dH = H - He0;
        W = We0 + dH / (He1 - He0) * (We1 - We0);
        dW = W - We0;
        *P1 = 2.0 * (He0 - b) + We0 + 2.0 * sqrt(pow(dH, 2) + pow(0.5*dW, 2));
        *P2 = 0.0;
    } else if (H < He2) {
        dH = H - He1;
        W = We1 + dH / (He2 - He1) * (We2 - We1);
        dW = W - We1;
        *P1 = 2.0 * (He0 - b) + We0 + 2.0 * sqrt(pow(He1 - He0, 2) + pow(We1 - We0, 2));
        *P2 = 2.0 * sqrt(pow(dH, 2) + pow(0.5*dW, 2));
    } else {
        dH = H - He2;
        W = We2 + dH / (He2 - He1) * (We2 - We1);
        dW = W - We2;
        *P1 = 2.0 * (He0 - b) + We0 + 2.0 * sqrt(pow(He1 - He0, 2) + pow(We1 - We0, 2));
        // A2 = (He2 - He1) * 0.5 * (We2 + We1) + dH * 0.5 * (W + We2) - (He2 - He1) * We1 - dH * We1
        // => A2 = (He2 - He1) * 0.5 * (We2 - We1) + dH * (0.5 * (W + We2) - We1)
        *P2 = 2.0 * sqrt(pow(He2 - He1, 2) + pow(0.5*(We2 - We1), 2)) + 2.0 * sqrt(pow(dH, 2) + pow(0.5*dW, 2));
    }
}

void critical_depth31(double Q,
                      double b,
                      double He0,
                      double He1,
                      double He2,
                      double We0,
                      double We1,
                      double We2,
                      double* dc) {

    uint16_t iter;
    double A;
    double dmin, dmax;
    double Fr2;
    double Q2;
    double W;

    Q2 = Q*Q;

    dmin = 0.1;
    dmax = He2 - b;

    // Test if hc > hmax
    width31(dmax+b, b, He0, He1, He2, We0, We1, We2, &W);
    area31(dmax+b, b, He0, He1, He2, We0, We1, We2, &A);
    Fr2 = Q2 * W / (CONSTANT_GRAVITY * std::pow(A, 3));
    while (Fr2 > 1.0) {
        dmax = 2.0 * dmax;
        width31(dmax+b, b, He0, He1, He2, We0, We1, We2, &W);
        area31(dmax+b, b, He0, He1, He2, We0, We1, We2, &A);
        Fr2 = Q2 * W / (CONSTANT_GRAVITY * std::pow(A, 3));
    }
    iter = 0;
    while (dmax - dmin > 1e-3 && iter < 1000) {
        iter++;
        *dc = 0.5 * (dmin + dmax);
        width31(*dc+b, b, He0, He1, He2, We0, We1, We2, &W);
        area31(*dc+b, b, He0, He1, He2, We0, We1, We2, &A);
        Fr2 = Q2 * W / (CONSTANT_GRAVITY * std::pow(A, 3));
        if (1.0 - Fr2 > 0.0) {
            dmax = *dc;
        } else {
            dmin = *dc;
        }
    }

}
