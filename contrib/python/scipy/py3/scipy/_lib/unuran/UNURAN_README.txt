
 *****************************************************************************
 *                                                                           *
 *          UNU.RAN -- Universal Non-Uniform Random number generator         *
 *                                                                           *
 *****************************************************************************


UNU.RAN is an ANSI C library.

Note: the following version was re-licensed to the BSD license of SciPy.
See license.txt for more details.

It contains universal (also called automatic or black-box) algorithms
that can generate random numbers from large classes of continuous or
discrete distributions, and also from practically all standard
distributions.

The library and an extensive online documentation are available at:

          ------------------------------------------
               http://statmath.wu.ac.at/unuran/ 
          ------------------------------------------


---------------------------------------------------------
A short overview
---------------------------------------------------------

To generate random numbers the user must supply some information about
the desired distribution, especially a C-function that computes the
density and - depending on the chosen methods - some additional
information (like the borders of the domain, the mode, the derivative
of the density ...). After a user has given this information an
init-program computes all tables and constants necessary for the
random variate generation. The sample program can then generate
variates from the desired distribution.

The main part 
   of UNU.RAN are different universal algorithms (called methods).
   There are:

	 7 methods for continuous univariate distributions
	 3 methods for discrete univariate distributions
	 1 method  for univariate empirical distributions
		   (given by an observed sample)
	 1 method  for multivariate empirical distributions
		   (given by a vector sample)

   The choice of the method depends on the information available for
   the distribution and on the desired characteristics of the algorithm
   (fast initialisation and slow sampling, slow initialisation and
   fast sampling).

A second important part
   of UNU.RAN is the distribution module containing all necessary
   functions for many continuous and discrete univariate standard
   distributions. Thus UNU.RAN can be used  without extra coding to
   obtain very fast generators for the best known standard
   distributions.


UNU.RAN is coded in ANSI C but uses an object oriented programming
interface. There are three objects:

   distribution objects
	containing all information of the distribution.
   parameter objects
	containing all input parameters (and defaults) for the
        different methods.
   generator object
	produced by the initialization program, containing everything
	necessary for sampling.


Of course a uniform random number generator is necessary to use
UNU.RAN. We provide an interface to use the PRNG uniform package from
the pLab team from the University of Salzburg (Austria),
available at http://random.mat.ac.at/ or from
http://statmath.wu.ac.at/prng/.
It is also no problem to include any other uniform random number
generator.


---------------------------------------------------------
ADVANTAGES OF UNUNRAN
---------------------------------------------------------

Why can it be worth the time to download UNU.RAN and to understand the
concept of its interface? Isn't it much faster to implement a simple
standard method for the distribution I am interested in?

- The first and main advantage lies in the modelling flexibility you
  gain for your simulation. Once you have installed UNU.RAN you can
  sample from practically all uni-modal (and other) distributions
  without coding more than the density functions. For a big number of
  standard distributions (and truncated versions of these standard
  distributions) you need not even code the densities as these are
  already included in UNU.RAN.

- It is possible to sample from non-standard distribution. In fact
  only a pointer to a function that returns e.g. the density at a
  given point x is required.

- Distributions can be exchanged easily. For example it is not
  difficult at all to start your simulation with the normal
  distribution, and switch to an empirical distribution later.

- The library contains reliable and fast generation algorithms. The
  characteristics of some these algorithms (like speed, expected
  number of uniforms required etc, ...) are only slightly influenced
  by the chosen distribution. (However numerical inversion is included
  as a (very slow) brute force algorithm for the rare cases where the
  more sophisticated methods do not work.)

- Correlation induction facilities are included. 

---------------------------------------------------------

March 31st, 2001

Josef Leydold     (leydold@statmath.wu.ac.at)
Wolfgang Hoermann (hormannw@boun.edu.tr)







