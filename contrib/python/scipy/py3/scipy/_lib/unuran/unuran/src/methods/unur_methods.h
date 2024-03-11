/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unur_methods.h                                                    *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines bitmasks to identify used method in generator objects     *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2022 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *   SPDX-License-Identifier: BSD-3-Clause                                   *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/
#ifndef UNUR_METHODS_H_SEEN
#define UNUR_METHODS_H_SEEN
/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**  Bitmask to indicate methods                                            **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/
/* bitmasks                                                                  */

#define UNUR_MASK_TYPE     0xff000000u   /* indicate type of method           */

/* discrete distributions */
#define UNUR_METH_DISCR    0x01000000u

#define UNUR_METH_DARI     0x01000001u
#define UNUR_METH_DAU      0x01000002u
#define UNUR_METH_DGT      0x01000003u
#define UNUR_METH_DSROU    0x01000004u
#define UNUR_METH_DSS      0x01000005u

/* continuous distributions */
#define UNUR_METH_CONT     0x02000000u

#define UNUR_METH_AROU     0x02000100u
#define UNUR_METH_ARS      0x02000d00u
#define UNUR_METH_HINV     0x02000200u
#define UNUR_METH_HRB      0x02000300u
#define UNUR_METH_HRD      0x02000400u
#define UNUR_METH_HRI      0x02000500u
#define UNUR_METH_ITDR     0x02000800u
#define UNUR_METH_NINV     0x02000600u
#define UNUR_METH_NROU     0x02000700u
#define UNUR_METH_PINV     0x02001000u
#define UNUR_METH_SROU     0x02000900u
#define UNUR_METH_SSR      0x02000a00u
#define UNUR_METH_TABL     0x02000b00u
#define UNUR_METH_TDR      0x02000c00u
#define UNUR_METH_UNIF     0x02000e00u
#define UNUR_METH_UTDR     0x02000f00u

/* univariate continuous empirical distributions */
#define UNUR_METH_CEMP     0x04000000u

#define UNUR_METH_EMPK     0x04001100u
#define UNUR_METH_EMPL     0x04001200u
#define UNUR_METH_HIST     0x04001300u

/* multivariate continuous distributions */
#define UNUR_METH_VEC      0x08000000u

#define UNUR_METH_MVTDR    0x08010000u
#define UNUR_METH_VMT      0x08020000u
#define UNUR_METH_VNROU    0x08030000u
#define UNUR_METH_VAROU    0x08040000u
#define UNUR_METH_NORTA    0x08050000u

#define UNUR_METH_GIBBS    0x08060000u
#define UNUR_METH_HITRO    0x08070000u
#define UNUR_METH_BALL     0x08080000u
#define UNUR_METH_WALK     0x08090000u

/* multivariate continuous empirical distributions */
#define UNUR_METH_CVEMP    0x10000000u

#define UNUR_METH_VEMPK    0x10010000u

/* random matrices */
#define UNUR_METH_MAT      0x20000000u

#define UNUR_METH_MCORR    0x20010000u

/* generators for standard distributions */
#define UNUR_METH_CSTD     0x0200f100u   /* is of type UNUR_METH_CONT !!     */
#define UNUR_METH_DSTD     0x0100f200u   /* is of type UNUR_METH_DISCR !!    */
#define UNUR_METH_MVSTD    0x0800f300u   /* is of type UNUR_METH_CVEC !!     */

/* meta distributions */
#define UNUR_METH_MIXT     0x0200e100u   /* univariate continuous            */

/* wrapper for external generators */
#define UNUR_METH_CEXT     0x0200f400u   /* is of type UNUR_METH_CONT !!     */
#define UNUR_METH_DEXT     0x0100f500u   /* is of type UNUR_METH_DISCR !!    */

/* automatically selected generator */
#define UNUR_METH_AUTO     0x00a00000u   /* can be any type of distribution  */

/* to indicate unknown type */
#define UNUR_METH_UNKNOWN  0xff000000u

/*---------------------------------------------------------------------------*/
#endif  /* UNUR_METHODS_H_SEEN */
/*---------------------------------------------------------------------------*/
