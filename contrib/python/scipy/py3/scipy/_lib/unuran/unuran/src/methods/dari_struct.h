/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: dari_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for method DARI                               *
 *         ((Discrete) Alias-Urn)                                            *
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

struct unur_dari_par { 
  int     squeeze;       /* should the squeeze be used  
                            0.. no squeeze,  1..squeeze                      */
  int     size;          /* size of table for speeding up generation         */
  double  c_factor;      /* constant for choosing the design points          */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_dari_gen { 
  double  vt;            /* total volume below hat                           */
  double  vc;            /* volume below center part                         */
  double  vcr;           /* volume center and right together                 */

  double  xsq[2];        /* value necesary for the squeeze computation       */
  double  y[2];          /* value of the transformed density in points of contact */
  double  ys[2];         /* the slope of the transformed hat                 */
  double  ac[2];         /* left and right starting point of the uniform hat 
                            in the center                                    */

  double  pm;            /* mode probability                                 */
  double  Hat[2];        /* point where the hat starts for the left and
                            the right tail                                   */
  double  c_factor;      /* constant for choosing the design points          */

  int     m;             /* mode                                             */
  int     x[2];          /* points of contact left and right of the mode     */
  int     s[2];          /* first and last integer of the center part        */
  int     n[2];          /* contains the first and the last i 
                            for which values are stored in table             */
  int     size;          /* size of the auxiliary tables                     */
  int     squeeze;       /* use squeeze yes/no                               */

  double *hp;            /* pointer to double array of length size           */
  char   *hb;            /* pointer to boolean array of length size          */
};

/*---------------------------------------------------------------------------*/


