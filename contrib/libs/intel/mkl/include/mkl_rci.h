/*******************************************************************************
* Copyright 2005-2022 Intel Corporation.
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

/*
!  Content:
!   Intel(R) oneAPI Math Kernel Library (oneMKL) interface for preconditioners, RCI ISS and
!   TR solvers routines
!******************************************************************************/

#ifndef _MKL_RCISOLVER_H_
#define _MKL_RCISOLVER_H_

#include "mkl_types.h"
#include "mkl_service.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern void dcsrilu0(const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, double *alu, const MKL_INT *ipar, const double *dpar,MKL_INT *ierr);
extern void dcsrilut(const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, double *alut, MKL_INT *ialut, MKL_INT *jalut, const double * tol, const MKL_INT *maxfil, const MKL_INT *ipar, const double *dpar,MKL_INT *ierr);

extern void DCSRILU0(const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, double *alu, const MKL_INT *ipar, const double *dpar,MKL_INT *ierr);
extern void DCSRILUT(const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, double *alut, MKL_INT *ialut, MKL_INT *jalut, const double * tol, const MKL_INT *maxfil, const MKL_INT *ipar, const double *dpar,MKL_INT *ierr);

/* PCG/PFGMRES Lower case */

extern void dcg_init(const MKL_INT *n, const double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcg_check(const MKL_INT *n, const double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcg(const MKL_INT *n, double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcg_get(const MKL_INT *n, const double *x, const double *b, const MKL_INT *rci_request, const MKL_INT *ipar, const double *dpar, const double *tmp, MKL_INT *itercount);

extern void dcgmrhs_init(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b,const  MKL_INT *method, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcgmrhs_check(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcgmrhs(const MKL_INT *n, double *x, const MKL_INT* nRhs, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dcgmrhs_get(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b, const MKL_INT *rci_request, const MKL_INT *ipar, const double *dpar, const double *tmp, MKL_INT *itercount);

extern void dfgmres_init(const MKL_INT *n, const double *x, const double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dfgmres_check(const MKL_INT *n, const double *x, const double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dfgmres(const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void dfgmres_get(const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, const MKL_INT *ipar, const double *dpar, double *tmp, MKL_INT *itercount);

/* PCG/PFGMRES Upper case */

extern void DCG_INIT(const MKL_INT *n, const double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCG_CHECK(const MKL_INT *n, const double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCG(const MKL_INT *n, double *x, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCG_GET(const MKL_INT *n, const double *x, const double *b, const MKL_INT *rci_request, const MKL_INT *ipar, const double *dpar, const double *tmp, MKL_INT *itercount);

extern void DCGMRHS_INIT(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b, const MKL_INT *method, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCGMRHS_CHECK(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCGMRHS(const MKL_INT *n, double *x, const MKL_INT* nRhs, const double *b, MKL_INT *rci_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DCGMRHS_GET(const MKL_INT *n, const double *x, const MKL_INT* nRhs, const double *b, const MKL_INT *rci_request, const MKL_INT *ipar, const double *dpar, const double *tmp, MKL_INT *itercount);

extern void DFGMRES_INIT(const MKL_INT *n, const double *x, const double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DFGMRES_CHECK(const MKL_INT *n, const double *x, const double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DFGMRES(const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, MKL_INT *ipar, double *dpar, double *tmp);
extern void DFGMRES_GET(const MKL_INT *n, double *x, double *b, MKL_INT *RCI_request, const MKL_INT *ipar, const double *dpar, double *tmp, MKL_INT *itercount);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* Return status values */
#define TR_SUCCESS        1501
#define TR_INVALID_OPTION 1502
#define TR_OUT_OF_MEMORY  1503

/* Basic data types */
typedef void* _TRNSP_HANDLE_t;
typedef void* _TRNSPBC_HANDLE_t;
typedef void* _JACOBIMATRIX_HANDLE_t;

typedef void(*USRFCND) (MKL_INT*,MKL_INT*,double*,double*);
typedef void(*USRFCNXD) (MKL_INT*,MKL_INT*,double*,double*,void*);

typedef void(*USRFCNS) (MKL_INT*,MKL_INT*,float*,float*);
typedef void(*USRFCNXS) (MKL_INT*,MKL_INT*,float*,float*,void*);

/* Function prototypes */
extern MKL_INT dtrnlsp_init     (_TRNSP_HANDLE_t*, const MKL_INT*, const MKL_INT*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*);
extern MKL_INT dtrnlsp_check    (_TRNSP_HANDLE_t*, const MKL_INT*, const MKL_INT*, const double*, const double*, const double*, MKL_INT*);
extern MKL_INT dtrnlsp_solve    (_TRNSP_HANDLE_t*, double*, double*, MKL_INT*);
extern MKL_INT dtrnlsp_get      (_TRNSP_HANDLE_t*, MKL_INT*, MKL_INT*, double*, double*);
extern MKL_INT dtrnlsp_delete   (_TRNSP_HANDLE_t*);

extern MKL_INT dtrnlspbc_init   (_TRNSPBC_HANDLE_t*, const MKL_INT*, const MKL_INT*, const double*, const double*, const double*, const double*, const MKL_INT*, const MKL_INT*, const double*);
extern MKL_INT dtrnlspbc_check  (_TRNSPBC_HANDLE_t*, const MKL_INT*, const MKL_INT*, const double*, const double*, const double*, const double*, const double*, MKL_INT*);
extern MKL_INT dtrnlspbc_solve  (_TRNSPBC_HANDLE_t*, double*, double*, MKL_INT*);
extern MKL_INT dtrnlspbc_get    (_TRNSPBC_HANDLE_t*, MKL_INT*, MKL_INT*, double*, double*);
extern MKL_INT dtrnlspbc_delete (_TRNSPBC_HANDLE_t*);

extern MKL_INT djacobi_init     (_JACOBIMATRIX_HANDLE_t*, const MKL_INT*, const MKL_INT*, const double*, const double*, const double*);
extern MKL_INT djacobi_solve    (_JACOBIMATRIX_HANDLE_t*, double*, double*, MKL_INT*);
extern MKL_INT djacobi_delete   (_JACOBIMATRIX_HANDLE_t*);
extern MKL_INT djacobi          (USRFCND fcn, const MKL_INT*, const MKL_INT*, double*, double*, double*);
extern MKL_INT djacobix         (USRFCNXD fcn, const MKL_INT*, const MKL_INT*, double*, double*, double*, void*);

extern MKL_INT strnlsp_init     (_TRNSP_HANDLE_t*, const MKL_INT*, const MKL_INT*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*);
extern MKL_INT strnlsp_check    (_TRNSP_HANDLE_t*, const MKL_INT*, const MKL_INT*, const float*, const float*, const float*, MKL_INT*);
extern MKL_INT strnlsp_solve    (_TRNSP_HANDLE_t*, float*, float*, MKL_INT*);
extern MKL_INT strnlsp_get      (_TRNSP_HANDLE_t*, MKL_INT*, MKL_INT*, float*, float*);
extern MKL_INT strnlsp_delete   (_TRNSP_HANDLE_t*);

extern MKL_INT strnlspbc_init   (_TRNSPBC_HANDLE_t*, const MKL_INT*, const MKL_INT*, const float*, const float*, const float*, const float*, const MKL_INT*, const MKL_INT*, const float*);
extern MKL_INT strnlspbc_check  (_TRNSPBC_HANDLE_t*, const MKL_INT*, const MKL_INT*, const float*, const float*, const float*, const float*, const float*, MKL_INT*);
extern MKL_INT strnlspbc_solve  (_TRNSPBC_HANDLE_t*, float*, float*, MKL_INT*);
extern MKL_INT strnlspbc_get    (_TRNSPBC_HANDLE_t*, MKL_INT*, MKL_INT*, float*, float*);
extern MKL_INT strnlspbc_delete (_TRNSPBC_HANDLE_t*);

extern MKL_INT sjacobi_init     (_JACOBIMATRIX_HANDLE_t*, const MKL_INT*, const MKL_INT*, const float*, const float*, const float*);
extern MKL_INT sjacobi_solve    (_JACOBIMATRIX_HANDLE_t*, float*, float*, MKL_INT*);
extern MKL_INT sjacobi_delete   (_JACOBIMATRIX_HANDLE_t*);
extern MKL_INT sjacobi          (USRFCNS fcn, const MKL_INT*, const MKL_INT*, float*, float*, float*);
extern MKL_INT sjacobix         (USRFCNXS fcn, const MKL_INT*, const MKL_INT*, float*, float*, float*, void*);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_RCISOLVER_H_ */
