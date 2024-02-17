#include "f2c.h"

int splev_(doublereal *t, integer *n, doublereal *c__, 
	integer *k, doublereal *x, doublereal *y, integer *m, integer *ier);

//(knots, coefficients [both of length num_knots], degree) represent the b-spline
//(this tuple is commonly referred to in scipy examples as t,c,k.)
//
//returns 0 if OK, non-0 if error
extern "C" int splev(const double* knots, int num_knots, const double* coefficients, int degree,
                     const double* positions_to_evaluate, double* spline_values, int num_spline_values)
{
    int ier = 0;
    splev_(const_cast<double*>(knots), &num_knots, const_cast<double*>(coefficients), &degree, const_cast<double*>(positions_to_evaluate), spline_values, &num_spline_values, &ier);
    return ier;
}
