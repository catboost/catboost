/* Ordered set data type implemented by a binary tree.
   Copyright (C) 2006-2007, 2009-2020 Free Software Foundation, Inc.
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
#include "gl_rbtree_oset.h"

#include <stdlib.h>

/* -------------------------- gl_oset_t Data Type -------------------------- */

/* Parameterization of gl_rbtree_ordered.h.  */
#define CONTAINER_T gl_oset_t
#define CONTAINER_IMPL gl_oset_impl
#define CONTAINER_IMPL_BASE gl_oset_impl_base
#define NODE_IMPL gl_oset_node_impl
#define NODE_T gl_oset_node_t
#define NODE_PAYLOAD_FIELDS \
  const void *value;
#define NODE_PAYLOAD_PARAMS \
  const void *elt
#define NODE_PAYLOAD_ASSIGN(node) \
  node->value = elt;
#define NODE_PAYLOAD_DISPOSE(container, node) \
  if (container->base.dispose_fn != NULL) \
    container->base.dispose_fn (node->value);

#include "gl_rbtree_ordered.h"

/* Generic binary tree code.  */
#include "gl_anytree_oset.h"

/* For debugging.  */
void
gl_rbtree_oset_check_invariants (gl_oset_t set)
{
  size_t counter = 0;
  if (set->root != NULL)
    check_invariants (set->root, NULL, &counter);
  if (!(set->count == counter))
    abort ();
}

const struct gl_oset_implementation gl_rbtree_oset_implementation =
  {
    gl_tree_nx_create_empty,
    gl_tree_size,
    gl_tree_search,
    gl_tree_search_atleast,
    gl_tree_nx_add,
    gl_tree_remove,
    gl_tree_update,
    gl_tree_oset_free,
    gl_tree_iterator,
    gl_tree_iterator_atleast,
    gl_tree_iterator_next,
    gl_tree_iterator_free
  };
