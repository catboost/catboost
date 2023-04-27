#ifndef CONTOURPY_MPL_2005_ORIGINAL_H
#define CONTOURPY_MPL_2005_ORIGINAL_H

#include "common.h"

namespace contourpy {

/* the data about edges, zones, and points -- boundary or not, exists
 * or not, z value 0, 1, or 2 -- is kept in a mesh sized data array */
typedef short Cdata;

/* information to decide on correct contour direction in saddle zones
 * is stored in a mesh sized array.  Only those entries corresponding
 * to saddle zones have nonzero values in this array. */
typedef char Saddle;

/* here is the minimum structure required to tell where we are in the
 * mesh sized data array */
struct Csite
{
    long edge;                  /* ij of current edge */
    long left;                  /* +-1 or +-imax as the zone is to right, left, below,
                                 * or above the edge */
    long imax;                  /* imax for the mesh */
    long jmax;                  /* jmax for the mesh */
    long n;                     /* number of points marked on this curve so far */
    long count;                 /* count of start markers visited */
    double zlevel[2];           /* contour levels, zlevel[1]<=zlevel[0]
                                 * signals single level case */
    Saddle *saddle;             /* saddle zone information for the mesh */
    char *reg;                  /* region array for the mesh (was int) */
    Cdata *data;                /* added by EF */
    long edge0, left0;          /* starting site on this curve for closure */
    int level0;                 /* starting level for closure */
    long edge00;                /* site needing START_ROW mark */

    /* making the actual marks requires a bunch of other stuff */
    const double *x, *y, *z;    /* mesh coordinates and function values */
    double *xcp, *ycp;          /* output contour points */
    short *kcp;                 /* kind of contour point */

    long i_chunk_size, j_chunk_size;
};

Csite *cntr_new();

void cntr_init(Csite *site, long iMax, long jMax, const double *x, const double *y,
               const double *z, const bool *mask, long i_chunk_size, long j_chunk_size);

void cntr_del(Csite *site);

py::tuple cntr_trace(Csite *site, double levels[], int nlevels);

} // namespace contourpy

#endif // CONTOURPY_MPL_2005_ORIGINAL_H
