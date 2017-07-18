#pragma once

/** Points to the first prime number in the sorted prime number table. */
extern const unsigned long* const _y_first_prime;

/** Points to the last prime number in the sorted prime number table.
 * Note that it is pointing not *past* the last element, but *at* the last element. */
extern const unsigned long* const _y_last_prime;

/**
 * Calculates the number of buckets for the hash table that will hold the given
 * number of elements.
 *
 * @param elementCount                  Number of elements that the hash table will hold.
 * @returns                             Number of buckets, a prime number that is
 *                                      greater or equal to `elementCount`.
 */
unsigned long HashBucketCount(unsigned long elementCount);
