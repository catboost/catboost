#pragma once

/*
 * thread-safe random number generator.
 *
 * specialized for:
 *  all unsigned types (return value in range [0, MAX_VALUE_FOR_TYPE])
 *  bool
 *  long double (return value in range [0, 1))
 *  double (return value in range [0, 1))
 *  float (return value in range [0, 1))
 */
template <class T>
T RandomNumber();

/*
 * returns value in range [0, max)
 */
template <class T>
T RandomNumber(T max);

/*
 * Re-initialize random state - useful after forking in multi-process programs.
 */
void ResetRandomState();

/*
 * Set random SEED
 */
void SetRandomSeed(int seed);
