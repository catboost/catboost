/* Sequential list data type implemented by a hash table with a binary tree.
   Copyright (C) 2006, 2008-2020 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2006.

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

#include <config.h>

/* Specification.  */
#include "gl_rbtreehash_list.h"

#include <stdint.h> /* for uintptr_t, SIZE_MAX */
#include <stdlib.h>

#include "gl_rbtree_oset.h"
#include "xsize.h"

#define WITH_HASHTABLE 1

/* Which kind of binary trees to use for ordered sets.  Quite arbitrary.  */
#define OSET_TREE_FLAVOR GL_RBTREE_OSET

/* -------------------------- gl_list_t Data Type -------------------------- */

/* Generic hash-table code: Type definitions.  */
#include "gl_anyhash1.h"

/* Generic red-black tree code: Type definitions.  */
#include "gl_anyrbtree_list1.h"

/* Generic hash-table code: Low-level code.  */
#define CONTAINER_T gl_list_t
#define CONTAINER_COUNT(list) \
  ((list)->root != NULL ? (list)->root->branch_size : 0)
#include "gl_anyhash2.h"

/* Generic binary tree code: Type definitions.  */
#include "gl_anytree_list1.h"

/* Hash-table with binary tree code: Handling of hash buckets.  */
#include "gl_anytreehash_list1.h"

/* Generic red-black tree code: Insertion/deletion algorithms.  */
#include "gl_anyrbtree_list2.h"

/* Generic binary tree code: Functions taking advantage of the hash table.  */
#include "gl_anytreehash_list2.h"

/* Generic binary tree code: All other functions.  */
#include "gl_anytree_list2.h"

/* For debugging.  */
static unsigned int
check_invariants (gl_list_node_t node, gl_list_node_t parent)
{
  unsigned int left_blackheight =
    (node->left != NULL ? check_invariants (node->left, node) : 0);
  unsigned int right_blackheight =
    (node->right != NULL ? check_invariants (node->right, node) : 0);

  if (!(node->parent == parent))
    abort ();
  if (!(node->branch_size
        == (node->left != NULL ? node->left->branch_size : 0)
           + 1 + (node->right != NULL ? node->right->branch_size : 0)))
    abort ();
  if (!(node->color == BLACK || node->color == RED))
    abort ();
  if (parent == NULL && !(node->color == BLACK))
    abort ();
  if (!(left_blackheight == right_blackheight))
    abort ();

  return left_blackheight + (node->color == BLACK ? 1 : 0);
}
void
gl_rbtreehash_list_check_invariants (gl_list_t list)
{
  if (list->root != NULL)
    check_invariants (list->root, NULL);
}


const struct gl_list_implementation gl_rbtreehash_list_implementation =
  {
    gl_tree_nx_create_empty,
    gl_tree_nx_create,
    gl_tree_size,
    gl_tree_node_value,
    gl_tree_node_nx_set_value,
    gl_tree_next_node,
    gl_tree_previous_node,
    gl_tree_get_at,
    gl_tree_nx_set_at,
    gl_tree_search_from_to,
    gl_tree_indexof_from_to,
    gl_tree_nx_add_first,
    gl_tree_nx_add_last,
    gl_tree_nx_add_before,
    gl_tree_nx_add_after,
    gl_tree_nx_add_at,
    gl_tree_remove_node,
    gl_tree_remove_at,
    gl_tree_remove,
    gl_tree_list_free,
    gl_tree_iterator,
    gl_tree_iterator_from_to,
    gl_tree_iterator_next,
    gl_tree_iterator_free,
    gl_tree_sortedlist_search,
    gl_tree_sortedlist_search_from_to,
    gl_tree_sortedlist_indexof,
    gl_tree_sortedlist_indexof_from_to,
    gl_tree_sortedlist_nx_add,
    gl_tree_sortedlist_remove
  };
