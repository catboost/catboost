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

/* Common code of gl_avltree_oset.c and gl_rbtree_oset.c.  */

/* An item on the stack used for iterating across the elements.  */
typedef struct
{
  gl_oset_node_t node;
  bool rightp;
} iterstack_item_t;

/* A stack used for iterating across the elements.  */
typedef iterstack_item_t iterstack_t[MAXHEIGHT];

static gl_oset_t
gl_tree_nx_create_empty (gl_oset_implementation_t implementation,
                         gl_setelement_compar_fn compar_fn,
                         gl_setelement_dispose_fn dispose_fn)
{
  struct gl_oset_impl *set =
    (struct gl_oset_impl *) malloc (sizeof (struct gl_oset_impl));

  if (set == NULL)
    return NULL;

  set->base.vtable = implementation;
  set->base.compar_fn = compar_fn;
  set->base.dispose_fn = dispose_fn;
  set->root = NULL;
  set->count = 0;

  return set;
}

static size_t _GL_ATTRIBUTE_PURE
gl_tree_size (gl_oset_t set)
{
  return set->count;
}

/* Returns the next node in the tree, or NULL if there is none.  */
static inline gl_oset_node_t _GL_ATTRIBUTE_PURE
gl_tree_next_node (gl_oset_node_t node)
{
  if (node->right != NULL)
    {
      node = node->right;
      while (node->left != NULL)
        node = node->left;
    }
  else
    {
      while (node->parent != NULL && node->parent->right == node)
        node = node->parent;
      node = node->parent;
    }
  return node;
}

/* Returns the previous node in the tree, or NULL if there is none.  */
static inline gl_oset_node_t _GL_ATTRIBUTE_PURE
gl_tree_prev_node (gl_oset_node_t node)
{
  if (node->left != NULL)
    {
      node = node->left;
      while (node->right != NULL)
        node = node->right;
    }
  else
    {
      while (node->parent != NULL && node->parent->left == node)
        node = node->parent;
      node = node->parent;
    }
  return node;
}

static bool
gl_tree_search (gl_oset_t set, const void *elt)
{
  gl_setelement_compar_fn compar = set->base.compar_fn;
  gl_oset_node_t node;

  for (node = set->root; node != NULL; )
    {
      int cmp = (compar != NULL
                 ? compar (node->value, elt)
                 : (node->value > elt ? 1 :
                    node->value < elt ? -1 : 0));

      if (cmp < 0)
        node = node->right;
      else if (cmp > 0)
        node = node->left;
      else /* cmp == 0 */
        /* We have an element equal to ELT.  */
        return true;
    }
  return false;
}

static bool
gl_tree_search_atleast (gl_oset_t set,
                        gl_setelement_threshold_fn threshold_fn,
                        const void *threshold,
                        const void **eltp)
{
  gl_oset_node_t node;

  for (node = set->root; node != NULL; )
    {
      if (! threshold_fn (node->value, threshold))
        node = node->right;
      else
        {
          /* We have an element >= THRESHOLD.  But we need the leftmost such
             element.  */
          gl_oset_node_t found = node;
          node = node->left;
          for (; node != NULL; )
            {
              if (! threshold_fn (node->value, threshold))
                node = node->right;
              else
                {
                  found = node;
                  node = node->left;
                }
            }
          *eltp = found->value;
          return true;
        }
    }
  return false;
}

static gl_oset_node_t
gl_tree_search_node (gl_oset_t set, const void *elt)
{
  gl_setelement_compar_fn compar = set->base.compar_fn;
  gl_oset_node_t node;

  for (node = set->root; node != NULL; )
    {
      int cmp = (compar != NULL
                 ? compar (node->value, elt)
                 : (node->value > elt ? 1 :
                    node->value < elt ? -1 : 0));

      if (cmp < 0)
        node = node->right;
      else if (cmp > 0)
        node = node->left;
      else /* cmp == 0 */
        /* We have an element equal to ELT.  */
        return node;
    }
  return NULL;
}

static int
gl_tree_nx_add (gl_oset_t set, const void *elt)
{
  gl_setelement_compar_fn compar;
  gl_oset_node_t node = set->root;

  if (node == NULL)
    {
      if (gl_tree_nx_add_first (set, elt) == NULL)
        return -1;
      return true;
    }

  compar = set->base.compar_fn;

  for (;;)
    {
      int cmp = (compar != NULL
                 ? compar (node->value, elt)
                 : (node->value > elt ? 1 :
                    node->value < elt ? -1 : 0));

      if (cmp < 0)
        {
          if (node->right == NULL)
            {
              if (gl_tree_nx_add_after (set, node, elt) == NULL)
                return -1;
              return true;
            }
          node = node->right;
        }
      else if (cmp > 0)
        {
          if (node->left == NULL)
            {
              if (gl_tree_nx_add_before (set, node, elt) == NULL)
                return -1;
              return true;
            }
          node = node->left;
        }
      else /* cmp == 0 */
        return false;
    }
}

static bool
gl_tree_remove (gl_oset_t set, const void *elt)
{
  gl_oset_node_t node = gl_tree_search_node (set, elt);

  if (node != NULL)
    return gl_tree_remove_node (set, node);
  else
    return false;
}

static int
gl_tree_update (gl_oset_t set, const void *elt,
                void (*action) (const void * /*elt*/, void * /*action_data*/),
                void *action_data)
{
  /* Like gl_tree_remove, action (...), gl_tree_nx_add, except that we don't
     actually remove ELT.  */
  /* Remember the old node.  Don't free it.  */
  gl_oset_node_t old_node = gl_tree_search_node (set, elt);
  /* Invoke ACTION.  */
  action (elt, action_data);
  /* Determine where to put the node now.  */
  if (old_node != NULL)
    {
      if (set->count > 1)
        {
          gl_setelement_compar_fn compar = set->base.compar_fn;

          gl_oset_node_t prev_node = gl_tree_prev_node (old_node);
          gl_oset_node_t next_node = gl_tree_next_node (old_node);
          if (!(compar != NULL
                ? (prev_node == NULL || compar (prev_node->value, elt) < 0)
                  && (next_node == NULL || compar (next_node->value, elt) > 0)
                : (prev_node == NULL || prev_node->value < elt)
                  && (next_node == NULL || next_node->value > elt)))
            {
              /* old_node needs to move in the tree.  */
              gl_oset_node_t node;

              /* Remove the node from the tree.  Don't free it.  */
              gl_tree_remove_node_no_free (set, old_node);

              node = set->root;

              for (;;)
                {
                  int cmp = (compar != NULL
                             ? compar (node->value, elt)
                             : (node->value > elt ? 1 :
                                node->value < elt ? -1 : 0));

                  if (cmp < 0)
                    {
                      if (node->right == NULL)
                        {
                          gl_tree_add_node_after (set, node, old_node);
                          return true;
                        }
                      node = node->right;
                    }
                  else if (cmp > 0)
                    {
                      if (node->left == NULL)
                        {
                          gl_tree_add_node_before (set, node, old_node);
                          return true;
                        }
                      node = node->left;
                    }
                  else /* cmp == 0 */
                    {
                      /* Two elements are the same.  */
                      NODE_PAYLOAD_DISPOSE (set, old_node)
                      free (old_node);
                      return -1;
                    }
                }
            }
        }
    }
  return 0;
}

static void
gl_tree_oset_free (gl_oset_t set)
{
  /* Iterate across all elements in post-order.  */
  gl_oset_node_t node = set->root;
  iterstack_t stack;
  iterstack_item_t *stack_ptr = &stack[0];

  for (;;)
    {
      /* Descend on left branch.  */
      for (;;)
        {
          if (node == NULL)
            break;
          stack_ptr->node = node;
          stack_ptr->rightp = false;
          node = node->left;
          stack_ptr++;
        }
      /* Climb up again.  */
      for (;;)
        {
          if (stack_ptr == &stack[0])
            goto done_iterate;
          stack_ptr--;
          node = stack_ptr->node;
          if (!stack_ptr->rightp)
            break;
          /* Free the current node.  */
          if (set->base.dispose_fn != NULL)
            set->base.dispose_fn (node->value);
          free (node);
        }
      /* Descend on right branch.  */
      stack_ptr->rightp = true;
      node = node->right;
      stack_ptr++;
    }
 done_iterate:
  free (set);
}

/* --------------------- gl_oset_iterator_t Data Type --------------------- */

static gl_oset_iterator_t _GL_ATTRIBUTE_PURE
gl_tree_iterator (gl_oset_t set)
{
  gl_oset_iterator_t result;
  gl_oset_node_t node;

  result.vtable = set->base.vtable;
  result.set = set;
  /* Start node is the leftmost node.  */
  node = set->root;
  if (node != NULL)
    while (node->left != NULL)
      node = node->left;
  result.p = node;
  /* End point is past the rightmost node.  */
  result.q = NULL;
#if defined GCC_LINT || defined lint
  result.i = 0;
  result.j = 0;
  result.count = 0;
#endif

  return result;
}

static gl_oset_iterator_t
gl_tree_iterator_atleast (gl_oset_t set,
                          gl_setelement_threshold_fn threshold_fn,
                          const void *threshold)
{
  gl_oset_iterator_t result;
  gl_oset_node_t node;

  result.vtable = set->base.vtable;
  result.set = set;
  /* End point is past the rightmost node.  */
  result.q = NULL;
#if defined GCC_LINT || defined lint
  result.i = 0;
  result.j = 0;
  result.count = 0;
#endif

  for (node = set->root; node != NULL; )
    {
      if (! threshold_fn (node->value, threshold))
        node = node->right;
      else
        {
          /* We have an element >= THRESHOLD.  But we need the leftmost such
             element.  */
          gl_oset_node_t found = node;
          node = node->left;
          for (; node != NULL; )
            {
              if (! threshold_fn (node->value, threshold))
                node = node->right;
              else
                {
                  found = node;
                  node = node->left;
                }
            }
          result.p = found;
          return result;
        }
    }
  result.p = NULL;
  return result;
}

static bool
gl_tree_iterator_next (gl_oset_iterator_t *iterator, const void **eltp)
{
  if (iterator->p != iterator->q)
    {
      gl_oset_node_t node = (gl_oset_node_t) iterator->p;
      *eltp = node->value;
      /* Advance to the next node.  */
      node = gl_tree_next_node (node);
      iterator->p = node;
      return true;
    }
  else
    return false;
}

static void
gl_tree_iterator_free (gl_oset_iterator_t *iterator _GL_ATTRIBUTE_MAYBE_UNUSED)
{
}
