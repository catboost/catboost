/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: mixt_struct.h                                                     *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         declares structures for meta method MIXT                          *
 *         (MIXTure of distributions)                                        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2010 Wolfgang Hoermann and Josef Leydold                  *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *                                                                           *

 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/* Information for constructing the generator                                */

struct unur_mixt_par { 
  int n_comp;                   /* number of components                      */
  const double *prob;           /* probabilities (weights) for components    */
  struct unur_gen **comp;       /* array of pointers to components           */
};

/*---------------------------------------------------------------------------*/
/* The generator object                                                      */

struct unur_mixt_gen { 
  int is_inversion;             /* whether inversion is used                 */

  /* components are stored in slot 'gen_aux_list'                            */
  /* probabilities are stored in slot 'gen_aux' as generator with method DGT */
};

/*---------------------------------------------------------------------------*/
