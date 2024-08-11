/* hash - hashing table processing.
   Copyright (C) 1998-1999, 2001, 2003, 2009-2020 Free Software Foundation,
   Inc.
   Written by Jim Meyering <meyering@ascend.com>, 1998.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* A generic hash table package.  */

/* Make sure USE_OBSTACK is defined to 1 if you want the allocator to use
   obstacks instead of malloc, and recompile 'hash.c' with same setting.  */

#ifndef HASH_H_
# define HASH_H_

# include <stdio.h>
# include <stdbool.h>

# ifdef __cplusplus
extern "C" {
# endif

struct hash_tuning
  {
    /* This structure is mainly used for 'hash_initialize', see the block
       documentation of 'hash_reset_tuning' for more complete comments.  */

    float shrink_threshold;     /* ratio of used buckets to trigger a shrink */
    float shrink_factor;        /* ratio of new smaller size to original size */
    float growth_threshold;     /* ratio of used buckets to trigger a growth */
    float growth_factor;        /* ratio of new bigger size to original size */
    bool is_n_buckets;          /* if CANDIDATE really means table size */
  };

typedef struct hash_tuning Hash_tuning;

struct hash_table;

typedef struct hash_table Hash_table;

/*
 * Information and lookup.
 */

/* The following few functions provide information about the overall hash
   table organization: the number of entries, number of buckets and maximum
   length of buckets.  */

/* Return the number of buckets in the hash table.  The table size, the total
   number of buckets (used plus unused), or the maximum number of slots, are
   the same quantity.  */
extern size_t hash_get_n_buckets (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

/* Return the number of slots in use (non-empty buckets).  */
extern size_t hash_get_n_buckets_used (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

/* Return the number of active entries.  */
extern size_t hash_get_n_entries (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

/* Return the length of the longest chain (bucket).  */
extern size_t hash_get_max_bucket_length (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

/* Do a mild validation of a hash table, by traversing it and checking two
   statistics.  */
extern bool hash_table_ok (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

extern void hash_print_statistics (const Hash_table *table, FILE *stream);

/* If ENTRY matches an entry already in the hash table, return the
   entry from the table.  Otherwise, return NULL.  */
extern void *hash_lookup (const Hash_table *table, const void *entry);

/*
 * Walking.
 */

/* The functions in this page traverse the hash table and process the
   contained entries.  For the traversal to work properly, the hash table
   should not be resized nor modified while any particular entry is being
   processed.  In particular, entries should not be added, and an entry
   may be removed only if there is no shrink threshold and the entry being
   removed has already been passed to hash_get_next.  */

/* Return the first data in the table, or NULL if the table is empty.  */
extern void *hash_get_first (const Hash_table *table)
       _GL_ATTRIBUTE_PURE;

/* Return the user data for the entry following ENTRY, where ENTRY has been
   returned by a previous call to either 'hash_get_first' or 'hash_get_next'.
   Return NULL if there are no more entries.  */
extern void *hash_get_next (const Hash_table *table, const void *entry);

/* Fill BUFFER with pointers to active user entries in the hash table, then
   return the number of pointers copied.  Do not copy more than BUFFER_SIZE
   pointers.  */
extern size_t hash_get_entries (const Hash_table *table, void **buffer,
                                size_t buffer_size);

typedef bool (*Hash_processor) (void *entry, void *processor_data);

/* Call a PROCESSOR function for each entry of a hash table, and return the
   number of entries for which the processor function returned success.  A
   pointer to some PROCESSOR_DATA which will be made available to each call to
   the processor function.  The PROCESSOR accepts two arguments: the first is
   the user entry being walked into, the second is the value of PROCESSOR_DATA
   as received.  The walking continue for as long as the PROCESSOR function
   returns nonzero.  When it returns zero, the walking is interrupted.  */
extern size_t hash_do_for_each (const Hash_table *table,
                                Hash_processor processor, void *processor_data);

/*
 * Allocation and clean-up.
 */

/* Return a hash index for a NUL-terminated STRING between 0 and N_BUCKETS-1.
   This is a convenience routine for constructing other hashing functions.  */
extern size_t hash_string (const char *string, size_t n_buckets)
       _GL_ATTRIBUTE_PURE;

extern void hash_reset_tuning (Hash_tuning *tuning);

typedef size_t (*Hash_hasher) (const void *entry, size_t table_size);
typedef bool (*Hash_comparator) (const void *entry1, const void *entry2);
typedef void (*Hash_data_freer) (void *entry);

/* Allocate and return a new hash table, or NULL upon failure.  The initial
   number of buckets is automatically selected so as to _guarantee_ that you
   may insert at least CANDIDATE different user entries before any growth of
   the hash table size occurs.  So, if have a reasonably tight a-priori upper
   bound on the number of entries you intend to insert in the hash table, you
   may save some table memory and insertion time, by specifying it here.  If
   the IS_N_BUCKETS field of the TUNING structure is true, the CANDIDATE
   argument has its meaning changed to the wanted number of buckets.

   TUNING points to a structure of user-supplied values, in case some fine
   tuning is wanted over the default behavior of the hasher.  If TUNING is
   NULL, the default tuning parameters are used instead.  If TUNING is
   provided but the values requested are out of bounds or might cause
   rounding errors, return NULL.

   The user-supplied HASHER function, when not NULL, accepts two
   arguments ENTRY and TABLE_SIZE.  It computes, by hashing ENTRY contents, a
   slot number for that entry which should be in the range 0..TABLE_SIZE-1.
   This slot number is then returned.

   The user-supplied COMPARATOR function, when not NULL, accepts two
   arguments pointing to user data, it then returns true for a pair of entries
   that compare equal, or false otherwise.  This function is internally called
   on entries which are already known to hash to the same bucket index,
   but which are distinct pointers.

   The user-supplied DATA_FREER function, when not NULL, may be later called
   with the user data as an argument, just before the entry containing the
   data gets freed.  This happens from within 'hash_free' or 'hash_clear'.
   You should specify this function only if you want these functions to free
   all of your 'data' data.  This is typically the case when your data is
   simply an auxiliary struct that you have malloc'd to aggregate several
   values.  */
extern Hash_table *hash_initialize (size_t candidate,
                                    const Hash_tuning *tuning,
                                    Hash_hasher hasher,
                                    Hash_comparator comparator,
                                    Hash_data_freer data_freer)
       _GL_ATTRIBUTE_NODISCARD;

/* Same as hash_initialize, but invokes xalloc_die on memory exhaustion.  */
/* This function is defined by module 'xhash'.  */
extern Hash_table *hash_xinitialize (size_t candidate,
                                     const Hash_tuning *tuning,
                                     Hash_hasher hasher,
                                     Hash_comparator comparator,
                                     Hash_data_freer data_freer)
       _GL_ATTRIBUTE_NODISCARD;

/* Make all buckets empty, placing any chained entries on the free list.
   Apply the user-specified function data_freer (if any) to the datas of any
   affected entries.  */
extern void hash_clear (Hash_table *table);

/* Reclaim all storage associated with a hash table.  If a data_freer
   function has been supplied by the user when the hash table was created,
   this function applies it to the data of each entry before freeing that
   entry.  */
extern void hash_free (Hash_table *table);

/*
 * Insertion and deletion.
 */

/* For an already existing hash table, change the number of buckets through
   specifying CANDIDATE.  The contents of the hash table are preserved.  The
   new number of buckets is automatically selected so as to _guarantee_ that
   the table may receive at least CANDIDATE different user entries, including
   those already in the table, before any other growth of the hash table size
   occurs.  If TUNING->IS_N_BUCKETS is true, then CANDIDATE specifies the
   exact number of buckets desired.  Return true iff the rehash succeeded.  */
extern bool hash_rehash (Hash_table *table, size_t candidate)
       _GL_ATTRIBUTE_NODISCARD;

/* If ENTRY matches an entry already in the hash table, return the pointer
   to the entry from the table.  Otherwise, insert ENTRY and return ENTRY.
   Return NULL if the storage required for insertion cannot be allocated.
   This implementation does not support duplicate entries or insertion of
   NULL.  */
extern void *hash_insert (Hash_table *table, const void *entry)
       _GL_ATTRIBUTE_NODISCARD;

/* Same as hash_insert, but invokes xalloc_die on memory exhaustion.  */
/* This function is defined by module 'xhash'.  */
extern void *hash_xinsert (Hash_table *table, const void *entry);

/* Insert ENTRY into hash TABLE if there is not already a matching entry.

   Return -1 upon memory allocation failure.
   Return 1 if insertion succeeded.
   Return 0 if there is already a matching entry in the table,
   and in that case, if MATCHED_ENT is non-NULL, set *MATCHED_ENT
   to that entry.

   This interface is easier to use than hash_insert when you must
   distinguish between the latter two cases.  More importantly,
   hash_insert is unusable for some types of ENTRY values.  When using
   hash_insert, the only way to distinguish those cases is to compare
   the return value and ENTRY.  That works only when you can have two
   different ENTRY values that point to data that compares "equal".  Thus,
   when the ENTRY value is a simple scalar, you must use
   hash_insert_if_absent.  ENTRY must not be NULL.  */
extern int hash_insert_if_absent (Hash_table *table, const void *entry,
                                  const void **matched_ent);

/* If ENTRY is already in the table, remove it and return the just-deleted
   data (the user may want to deallocate its storage).  If ENTRY is not in the
   table, don't modify the table and return NULL.  */
extern void *hash_remove (Hash_table *table, const void *entry);

/* Same as hash_remove.  This interface is deprecated.
   FIXME: Remove in 2022.  */
extern void *hash_delete (Hash_table *table, const void *entry)
       _GL_ATTRIBUTE_DEPRECATED;

# ifdef __cplusplus
}
# endif

#endif
