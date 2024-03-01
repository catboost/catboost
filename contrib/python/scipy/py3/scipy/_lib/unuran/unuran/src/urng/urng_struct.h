/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unur_struct.h                                                     *
 *                                                                           *
 *   structures used for included uniform random number generators           *
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

struct unur_urng {
  double (*sampleunif)(void *state);  /* function for generating uniform RNG */
  void *state;                        /* state of the generator              */
  unsigned int (*samplearray)(void *state, double *X, int dim);
                                      /* function for generating array of points */
  void (*sync)(void *state);          /* jump into defined state ("sync")    */
  unsigned long seed;                 /* initial seed of the generator       */
  void (*setseed)(void *state, unsigned long seed);  /* set initial seed of the generator */
  void (*reset)(void *state);         /* reset object                        */
  void (*nextsub)(void *state);       /* skip to next substream              */
  void (*resetsub)(void *state);      /* reset current substream             */
  void (*anti)(void *state, int a);   /* set antithetic flag                 */
  void (*delete)(void *state);        /* function for destroying URNG        */

#ifdef UNUR_COOKIES
  unsigned cookie;            /* magic cookie                                */
#endif
};

/* Remark:                                                                   */
/* The 'seed' is used to fill the state variable(s) of a generator with      */
/* initial values. For some generators the size of the 'state' is larger     */
/* than the size if the 'seed'. In such a case some function is used to      */
/* expand the 'seed' to the appropriate size (often this is done with a      */
/* LCG (linear congruental generator) with starting state 'seed'; e.g. this  */
/* happens for the Mersenne twister).                                        */
/* Often the seed is not stored in the structure for the generator object    */
/* (which is stored in 'state').                                             */
/* Thus we have introduced an additional field 'seed' to store the           */
/* starting value.                                                           */

/*---------------------------------------------------------------------------*/
