/* Sequential list data type implemented by a binary tree.
   Copyright (C) 2006, 2009-2020 Free Software Foundation, Inc.
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

/* Common code of gl_rbtree_list.c and gl_rbtreehash_list.c.  */

/* A red-black tree is a binary tree where every node is colored black or
   red such that
   1. The root is black.
   2. No red node has a red parent.
      Or equivalently: No red node has a red child.
   3. All paths from the root down to any NULL endpoint contain the same
      number of black nodes.
   Let's call this the "black-height" bh of the tree.  It follows that every
   such path contains exactly bh black and between 0 and bh red nodes.  (The
   extreme cases are a path containing only black nodes, and a path colored
   alternately black-red-black-red-...-black-red.)  The height of the tree
   therefore is >= bh, <= 2*bh.
 */

/* -------------------------- gl_list_t Data Type -------------------------- */

/* Color of a node.  */
typedef enum color { BLACK, RED } color_t;

/* Concrete list node implementation, valid for this file only.  */
struct gl_list_node_impl
{
#if WITH_HASHTABLE
  struct gl_hash_entry h;           /* hash table entry fields; must be first */
#endif
  struct gl_list_node_impl *left;   /* left branch, or NULL */
  struct gl_list_node_impl *right;  /* right branch, or NULL */
  /* Parent pointer, or NULL. The parent pointer is not needed for most
     operations.  It is needed so that a gl_list_node_t can be returned
     without memory allocation, on which the functions gl_list_remove_node,
     gl_list_add_before, gl_list_add_after can be implemented.  */
  struct gl_list_node_impl *parent;
  color_t color;                    /* node's color */
  size_t branch_size;               /* number of nodes in this branch,
                                       = branchsize(left)+branchsize(right)+1 */
  const void *value;
};

/* Concrete gl_list_impl type, valid for this file only.  */
struct gl_list_impl
{
  struct gl_list_impl_base base;
#if WITH_HASHTABLE
  /* A hash table: managed as an array of collision lists.  */
  struct gl_hash_entry **table;
  size_t table_size;
#endif
  struct gl_list_node_impl *root;   /* root node or NULL */
};

/* A red-black tree of height h has a black-height bh >= ceil(h/2) and
   therefore at least 2^ceil(h/2) - 1 elements.  So, h <= 116 (because a tree
   of height h >= 117 would have at least 2^59 - 1 elements, and because even
   on 64-bit machines,
     sizeof (gl_list_node_impl) * (2^59 - 1) > 2^64
   this would exceed the address space of the machine.  */
#define MAXHEIGHT 116
