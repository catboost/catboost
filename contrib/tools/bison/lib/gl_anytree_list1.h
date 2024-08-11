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

/* Common code of gl_avltree_list.c, gl_rbtree_list.c,
                  gl_avltreehash_list.c, gl_rbtreehash_list.c.  */

/* An item on the stack used for iterating across the elements.  */
typedef struct
{
  gl_list_node_t node;
  size_t rightp;
} iterstack_item_t;

/* A stack used for iterating across the elements.  */
typedef iterstack_item_t iterstack_t[MAXHEIGHT];

/* Frees a non-empty subtree recursively.
   This function is recursive and therefore not very fast.  */
static void
free_subtree (gl_list_node_t node)
{
  if (node->left != NULL)
    free_subtree (node->left);
  if (node->right != NULL)
    free_subtree (node->right);
  free (node);
}
