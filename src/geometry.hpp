#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

void area31(double H,
            double b,
            double He0,
            double He1,
            double He2,
            double We0,
            double We1,
            double We2,
            double* A);

void width31(double H,
             double b,
             double He0,
             double He1,
             double He2,
             double We0,
             double We1,
             double We2,
             double* W);

void compound_area31(double H,
                     double b,
                     double He0,
                     double He1,
                     double He2,
                     double We0,
                     double We1,
                     double We2,
                     double* A1,
                     double* A2);

void compound_perim31(double H,
                      double b,
                      double He0,
                      double He1,
                      double He2,
                      double We0,
                      double We1,
                      double We2,
                      double* P1,
                      double* P2);

void critical_depth31(double Q,
                      double b,
                      double He0,
                      double He1,
                      double He2,
                      double We0,
                      double We1,
                      double We2,
                      double* dc);

#endif