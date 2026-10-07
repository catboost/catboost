/*******************************************************************************
* Copyright 2007-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef _MKL_POISSON_H_
#define _MKL_POISSON_H_

/* definitions of Intel(R) oneAPI Math Kernel Library (oneMKL) types */
#include "mkl_types.h"
#include "mkl_dfti.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
#if defined( _WIN32 ) || ( _WIN64 )

#define   d_init_Helmholtz_2D   D_INIT_HELMHOLTZ_2D
#define d_commit_Helmholtz_2D D_COMMIT_HELMHOLTZ_2D
#define        d_Helmholtz_2D        D_HELMHOLTZ_2D
#define     free_Helmholtz_2D     FREE_HELMHOLTZ_2D

#define   d_init_Helmholtz_3D   D_INIT_HELMHOLTZ_3D
#define d_commit_Helmholtz_3D D_COMMIT_HELMHOLTZ_3D
#define        d_Helmholtz_3D        D_HELMHOLTZ_3D
#define     free_Helmholtz_3D     FREE_HELMHOLTZ_3D

#define   s_init_Helmholtz_2D   S_INIT_HELMHOLTZ_2D
#define s_commit_Helmholtz_2D S_COMMIT_HELMHOLTZ_2D
#define        s_Helmholtz_2D        S_HELMHOLTZ_2D

#define   s_init_Helmholtz_3D   S_INIT_HELMHOLTZ_3D
#define s_commit_Helmholtz_3D S_COMMIT_HELMHOLTZ_3D
#define        s_Helmholtz_3D        S_HELMHOLTZ_3D

#define   d_init_sph_p      D_INIT_SPH_P
#define d_commit_sph_p    D_COMMIT_SPH_P
#define        d_sph_p           D_SPH_P
#define     free_sph_p        FREE_SPH_P

#define   d_init_sph_np     D_INIT_SPH_NP
#define d_commit_sph_np   D_COMMIT_SPH_NP
#define        d_sph_np          D_SPH_NP
#define     free_sph_np       FREE_SPH_NP

#define   s_init_sph_p      S_INIT_SPH_P
#define s_commit_sph_p    S_COMMIT_SPH_P
#define        s_sph_p           S_SPH_P

#define   s_init_sph_np     S_INIT_SPH_NP
#define s_commit_sph_np   S_COMMIT_SPH_NP
#define        s_sph_np          S_SPH_NP

#else

#define   d_init_Helmholtz_2D    d_init_helmholtz_2d_
#define d_commit_Helmholtz_2D  d_commit_helmholtz_2d_
#define        d_Helmholtz_2D         d_helmholtz_2d_
#define     free_Helmholtz_2D      free_helmholtz_2d_

#define   d_init_Helmholtz_3D    d_init_helmholtz_3d_
#define d_commit_Helmholtz_3D  d_commit_helmholtz_3d_
#define        d_Helmholtz_3D         d_helmholtz_3d_
#define     free_Helmholtz_3D      free_helmholtz_3d_

#define   s_init_Helmholtz_2D    s_init_helmholtz_2d_
#define s_commit_Helmholtz_2D  s_commit_helmholtz_2d_
#define        s_Helmholtz_2D         s_helmholtz_2d_

#define   s_init_Helmholtz_3D    s_init_helmholtz_3d_
#define s_commit_Helmholtz_3D  s_commit_helmholtz_3d_
#define        s_Helmholtz_3D         s_helmholtz_3d_

#define   d_init_sph_p      d_init_sph_p_
#define d_commit_sph_p    d_commit_sph_p_
#define        d_sph_p           d_sph_p_
#define     free_sph_p        free_sph_p_

#define   d_init_sph_np   d_init_sph_np_
#define d_commit_sph_np d_commit_sph_np_
#define        d_sph_np        d_sph_np_
#define     free_sph_np     free_sph_np_

#define   s_init_sph_p      s_init_sph_p_
#define s_commit_sph_p    s_commit_sph_p_
#define        s_sph_p           s_sph_p_

#define   s_init_sph_np   s_init_sph_np_
#define s_commit_sph_np s_commit_sph_np_
#define        s_sph_np        s_sph_np_

#endif
**/

/**/
#define   d_init_Helmholtz_2D    d_init_helmholtz_2d
#define d_commit_Helmholtz_2D  d_commit_helmholtz_2d
#define        d_Helmholtz_2D         d_helmholtz_2d
#define     free_Helmholtz_2D      free_helmholtz_2d

#define   d_init_Helmholtz_3D    d_init_helmholtz_3d
#define d_commit_Helmholtz_3D  d_commit_helmholtz_3d
#define        d_Helmholtz_3D         d_helmholtz_3d
#define     free_Helmholtz_3D      free_helmholtz_3d

#define   s_init_Helmholtz_2D    s_init_helmholtz_2d
#define s_commit_Helmholtz_2D  s_commit_helmholtz_2d
#define        s_Helmholtz_2D         s_helmholtz_2d

#define   s_init_Helmholtz_3D    s_init_helmholtz_3d
#define s_commit_Helmholtz_3D  s_commit_helmholtz_3d
#define        s_Helmholtz_3D         s_helmholtz_3d
/**/

void   d_init_Helmholtz_2D(const double*, const double*, const double*, const double*, const MKL_INT*, const MKL_INT*, const char*, const double*, MKL_INT*, double*, MKL_INT*);
void d_commit_Helmholtz_2D(double*, const double*, const double*, const double*, const double*, DFTI_DESCRIPTOR_HANDLE *, MKL_INT*, double*,MKL_INT*);
void        d_Helmholtz_2D(double*, const double*, const double*, const double*, const double*, DFTI_DESCRIPTOR_HANDLE *, MKL_INT*, const double*,MKL_INT*);
void     free_Helmholtz_2D(DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, MKL_INT*);

void   d_init_Helmholtz_3D(const double*, const double*, const double*, const double*, const double*, const double*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const char*, const double*, MKL_INT*, double*, MKL_INT*);
void d_commit_Helmholtz_3D(double*, const double*, const double*, const double*, const double*, const double*, const double*, DFTI_DESCRIPTOR_HANDLE*, DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, double*, MKL_INT*);
void        d_Helmholtz_3D(double*, const double*, const double*, const double*, const double*, const double*, const double*, DFTI_DESCRIPTOR_HANDLE*, DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, const double*,MKL_INT*);
void     free_Helmholtz_3D(DFTI_DESCRIPTOR_HANDLE*, DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, MKL_INT*);

void   s_init_Helmholtz_2D(const float*, const float*, const float*, const float*, const MKL_INT*, const MKL_INT*, const char*, const float*, MKL_INT*, float*, MKL_INT*);
void s_commit_Helmholtz_2D(float*, const float*, const float*, const float*, const float*, DFTI_DESCRIPTOR_HANDLE *, MKL_INT*, float*, MKL_INT*);
void        s_Helmholtz_2D(float*, const float*, const float*, const float*, const float*, DFTI_DESCRIPTOR_HANDLE *, MKL_INT*, const float*, MKL_INT*);

void   s_init_Helmholtz_3D(const float*, const float*, const float*, const float*, const float*, const float*, const MKL_INT*, const MKL_INT*, const MKL_INT*, const char*, const float*,MKL_INT*,float*,MKL_INT*);
void s_commit_Helmholtz_3D(float*, const float*, const float*, const float*, const float*, const float*, const float*, DFTI_DESCRIPTOR_HANDLE*, DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, float*, MKL_INT*);
void        s_Helmholtz_3D(float*, const float*, const float*, const float*, const float*, const float*, const float*, DFTI_DESCRIPTOR_HANDLE*, DFTI_DESCRIPTOR_HANDLE*, MKL_INT*, const float*,MKL_INT*);

void   d_init_sph_p(const double*, const double*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*, MKL_INT*, double*, MKL_INT*);
void d_commit_sph_p(double*,DFTI_DESCRIPTOR_HANDLE*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,double*,MKL_INT*);
void        d_sph_p(double*,DFTI_DESCRIPTOR_HANDLE*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,double*,MKL_INT*);
void     free_sph_p(DFTI_DESCRIPTOR_HANDLE*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,MKL_INT*);

void   d_init_sph_np(const double*, const double*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*, MKL_INT*, double*, MKL_INT*);
void d_commit_sph_np(double*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,double*,MKL_INT*);
void        d_sph_np(double*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,double*,MKL_INT*);
void     free_sph_np(DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,MKL_INT*);

void   s_init_sph_p(const float*, const float*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*, MKL_INT*, float*,MKL_INT*);
void s_commit_sph_p(float*,DFTI_DESCRIPTOR_HANDLE*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,float*,MKL_INT*);
void        s_sph_p(float*,DFTI_DESCRIPTOR_HANDLE*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,float*,MKL_INT*);

void   s_init_sph_np(const float*, const float*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*, MKL_INT*, float*,MKL_INT*);
void s_commit_sph_np(float*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,float*,MKL_INT*);
void        s_sph_np(float*,DFTI_DESCRIPTOR_HANDLE*,MKL_INT*,float*,MKL_INT*);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_POISSON_H_ */
