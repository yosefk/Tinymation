/* fpbspl.f -- translated by f2c (version 20160102).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Subroutine */ int fpbspl_(doublereal *t, integer *n, integer *k, 
	doublereal *x, integer *l, doublereal *h__)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    doublereal f;
    integer i__, j;
    doublereal hh[5];
    integer li, lj;
    doublereal one;

/*  subroutine fpbspl evaluates the (k+1) non-zero b-splines of */
/*  degree k at t(l) <= x < t(l+1) using the stable recurrence */
/*  relation of de boor and cox. */
/*  .. */
/*  ..scalar arguments.. */
/*  ..array arguments.. */
/*  ..local scalars.. */
/*  ..local arrays.. */
/*  .. */
    /* Parameter adjustments */
    --t;
    --h__;

    /* Function Body */
    one = 1.;
    h__[1] = one;
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    hh[i__ - 1] = h__[i__];
/* L10: */
	}
	h__[1] = 0.;
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    li = *l + i__;
	    lj = li - j;
	    f = hh[i__ - 1] / (t[li] - t[lj]);
	    h__[i__] += f * (t[li] - *x);
	    h__[i__ + 1] = f * (*x - t[lj]);
/* L20: */
	}
    }
    return 0;
} /* fpbspl_ */

