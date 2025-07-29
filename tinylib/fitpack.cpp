#include <vector>
#include "f2c.h"

int splev_(doublereal *t, integer *n, doublereal *c__, 
	integer *k, doublereal *x, doublereal *y, integer *m, integer *ier);

//(knots, coefficients, spline_degree) represent the b-spline
//(this tuple is commonly referred to in scipy examples as t,c,k.)
//
//returns 0 if OK, non-0 if error
extern "C" int fitpack_splev(const double* knots, int num_knots, const double* coefficients, int spline_degree,
                             const double* positions_to_evaluate, double* spline_values, int num_spline_values)
{
    int ier = 0;
    splev_(const_cast<double*>(knots), &num_knots, const_cast<double*>(coefficients), &spline_degree, const_cast<double*>(positions_to_evaluate), spline_values, &num_spline_values, &ier);
    return ier;
}

int splder_(doublereal *t, integer *n, doublereal *c__, 
	integer *k, integer *nu, doublereal *x, doublereal *y, integer *m, 
	doublereal *wrk, integer *ier);

//like fitpack_splev but evaluates the Nth derivative, derivative_num=N
extern "C" int fitpack_splder(const double* knots, int num_knots, const double* coefficients, int spline_degree, int derivative_num,
                             const double* positions_to_evaluate, double* spline_values, int num_spline_values)
{
    int ier = 0;
    std::vector<double> wrk(num_knots); 
    splder_(const_cast<double*>(knots), &num_knots, const_cast<double*>(coefficients), &spline_degree, &derivative_num,
            const_cast<double*>(positions_to_evaluate), spline_values, &num_spline_values, &wrk[0], &ier);
    return ier;
}

//this comment is simplified for our use case - parcur.cpp has the "real" comment
//
//iopt: 0 for smoothing spline curve, -1/1 - we don't use
//ipar: 0 for parcur() to calculate u[i], ue and ub itself
//idim: curve dimension (1 to 10 inclusive, we fit a 2D curve)
//m: #points
//u: an array of dimension m, filled by parcur() with ipar=0
//mx: idim * m
//x: data points, array of size mx - x[idim*i+j] is coordinate j of data point i
//w: an array of m weights
//ub, ue: bounds on the value of u, filled automatically when ipar=0
//k: the degree of the spline (we use 3)
//s: smoothing factor
//nest: an upper bound on the number of knots returned by the function,
//      m+k+1 is always large enough
//n: the number of knots in the output curve (invalid when ier is 10)
//t: an array of dimension nest for the n output knots.
//nc: nest*idim
//c: an array of dimension nc for the output bspline coefficients.
//fp: weighted sum of squared residuals of the output bspline
//wrk: working space of m*(k+1)+nest*(6+idim+3*k) doubles
//lwrk: the dimension of wrk
//iwrk: working space of nest integers
//ier: 0 if OK, non-0 if error
int parcur_(integer *iopt, integer *ipar, integer *idim, 
	integer *m, doublereal *u, integer *mx, doublereal *x, doublereal *w, 
	doublereal *ub, doublereal *ue, integer *k, doublereal *s, integer *
	nest, integer *n, doublereal *t, integer *nc, doublereal *c__, 
	doublereal *fp, doublereal *wrk, integer *lwrk, integer *iwrk, 
	integer *ier);

//weights is of dimension num_points
//points is of dimension num_points * points_dimension
//knots is of dimension num_points + spline_degree + 1
//coefficients is of dimension (num_points + spline_degree + 1) * points_dimension
//u[i] is the value of the output bspline parameter producing the point closest to points[i]
extern "C" int fitpack_parcur(const double* points, const double* weights, int points_dimension, int num_points, double* u, int spline_degree,
                              double smoothing_factor, double* knots, int* num_knots, double* coefficients)
{
    int iopt = 0;
    int ipar = 0;
    int mx = points_dimension * num_points;
    double ub = 0;
    double ue = 0;
    int nest = max(2*spline_degree + 3, num_points + spline_degree + 1);
    int nc = nest * points_dimension;
    double fp = 0;
    int lwrk = num_points*(spline_degree+1)+nest*(6+points_dimension+3*spline_degree);
    std::vector<double> wrk(lwrk);
    std::vector<int> iwrk(nest);
    int ier = 0;

    parcur_(&iopt, &ipar, &points_dimension, &num_points, u, &mx, const_cast<double*>(points), const_cast<double*>(weights),
            &ub, &ue, &spline_degree, &smoothing_factor, &nest, num_knots, knots, &nc, coefficients,
            &fp, &wrk[0], &lwrk, &iwrk[0], &ier);

    return ier;
}
