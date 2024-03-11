/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: utdr_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method UTDR                               *
 *         (Universal Transformed Density Rejection; 3-point method)         *
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
/* Information for constructing the generator                                */

struct unur_utdr_par { 
  double  fm;               /* PDF at mode                                   */
  double  hm;               /* transformed PDF at mode                       */
  double  c_factor;         /* constant for choosing the design points       */
  double  delta_factor;     /* delta to replace the tangent                  */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_utdr_gen { 
  double  il;               /* left border of the domain                     */
  double  ir;               /* right border of the domain                    */
  double  fm;               /* PDF at mode                                   */
  double  hm;               /* transformed PDF at mode                       */

  double  vollc,volcompl,voll,
    al,ar,col,cor,sal,sar,bl,br,ttlx,ttrx,
    brblvolc,drar,dlal,ooar2,ooal2;/* constants of the hat and for generation*/

  double  c_factor;         /* constant for choosing the design points       */
  double  delta_factor;     /* delta to replace the tangent                  */
};

/*---------------------------------------------------------------------------*/
