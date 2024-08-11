/* Sequential list data type implemented by a binary tree.
   Copyright (C) 2006-2020 Free Software Foundation, Inc.
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

static gl_list_t
gl_tree_nx_create_empty (gl_list_implementation_t implementation,
                         gl_listelement_equals_fn equals_fn,
                         gl_listelement_hashcode_fn hashcode_fn,
                         gl_listelement_dispose_fn dispose_fn,
                         bool allow_duplicates)
{
  struct gl_list_impl *list = (struct gl_list_impl *) malloc (sizeof (struct gl_list_impl));

  if (list == NULL)
    return NULL;

  list->base.vtable = implementation;
  list->base.equals_fn = equals_fn;
  list->base.hashcode_fn = hashcode_fn;
  list->base.dispose_fn = dispose_fn;
  list->base.allow_duplicates = allow_duplicates;
#if WITH_HASHTABLE
  list->table_size = 11;
  list->table =
    (gl_hash_entry_t *) calloc (list->table_size, sizeof (gl_hash_entry_t));
  if (list->table == NULL)
    goto fail;
#endif
  list->root = NULL;

  return list;

#if WITH_HASHTABLE
 fail:
  free (list);
  return NULL;
#endif
}

static size_t _GL_ATTRIBUTE_PURE
gl_tree_size (gl_list_t list)
{
  return (list->root != NULL ? list->root->branch_size : 0);
}

static const void * _GL_ATTRIBUTE_PURE
gl_tree_node_value (gl_list_t list _GL_ATTRIBUTE_MAYBE_UNUSED,
                    gl_list_node_t node)
{
  return node->value;
}

static int
gl_tree_node_nx_set_value (gl_list_t list _GL_ATTRIBUTE_MAYBE_UNUSED,
                           gl_list_node_t node, const void *elt)
{
#if WITH_HASHTABLE
  if (elt != node->value)
    {
      size_t new_hashcode =
        (list->base.hashcode_fn != NULL
         ? list->base.hashcode_fn (elt)
         : (size_t)(uintptr_t) elt);

      if (new_hashcode != node->h.hashcode)
        {
          remove_from_bucket (list, node);
          node->value = elt;
          node->h.hashcode = new_hashcode;
          if (add_to_bucket (list, node) < 0)
            {
              /* Out of memory.  We removed node from a bucket but cannot add
                 it to another bucket.  In order to avoid inconsistencies, we
                 must remove node entirely from the list.  */
              gl_tree_remove_node_from_tree (list, node);
              free (node);
              return -1;
            }
        }
      else
        node->value = elt;
    }
#else
  node->value = elt;
#endif
  return 0;
}

static gl_list_node_t _GL_ATTRIBUTE_PURE
gl_tree_next_node (gl_list_t list _GL_ATTRIBUTE_MAYBE_UNUSED,
                   gl_list_node_t node)
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

static gl_list_node_t _GL_ATTRIBUTE_PURE
gl_tree_previous_node (gl_list_t list _GL_ATTRIBUTE_MAYBE_UNUSED,
                       gl_list_node_t node)
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

/* Returns the node at the given position < gl_tree_size (list).  */
static gl_list_node_t _GL_ATTRIBUTE_PURE
node_at (gl_list_node_t root, size_t position)
{
  /* Here we know that root != NULL.  */
  gl_list_node_t node = root;

  for (;;)
    {
      if (node->left != NULL)
        {
          if (position < node->left->branch_size)
            {
              node = node->left;
              continue;
            }
          position -= node->left->branch_size;
        }
      if (position == 0)
        break;
      position--;
      node = node->right;
    }
  return node;
}

static const void * _GL_ATTRIBUTE_PURE
gl_tree_get_at (gl_list_t list, size_t position)
{
  gl_list_node_t node = list->root;

  if (!(node != NULL && position < node->branch_size))
    /* Invalid argument.  */
    abort ();
  node = node_at (node, position);
  return node->value;
}

static gl_list_node_t
gl_tree_nx_set_at (gl_list_t list, size_t position, const void *elt)
{
  gl_list_node_t node = list->root;

  if (!(node != NULL && position < node->branch_size))
    /* Invalid argument.  */
    abort ();
  node = node_at (node, position);
#if WITH_HASHTABLE
  if (elt != node->value)
    {
      size_t new_hashcode =
        (list->base.hashcode_fn != NULL
         ? list->base.hashcode_fn (elt)
         : (size_t)(uintptr_t) elt);

      if (new_hashcode != node->h.hashcode)
        {
          remove_from_bucket (list, node);
          node->value = elt;
          node->h.hashcode = new_hashcode;
          if (add_to_bucket (list, node) < 0)
            {
              /* Out of memory.  We removed node from a bucket but cannot add
                 it to another bucket.  In order to avoid inconsistencies, we
                 must remove node entirely from the list.  */
              gl_tree_remove_node_from_tree (list, node);
              free (node);
              return NULL;
            }
        }
      else
        node->value = elt;
    }
#else
  node->value = elt;
#endif
  return node;
}

#if !WITH_HASHTABLE

static gl_list_node_t _GL_ATTRIBUTE_PURE
gl_tree_search_from_to (gl_list_t list, size_t start_index, size_t end_index,
                        const void *elt)
{
  if (!(start_index <= end_index
        && end_index <= (list->root != NULL ? list->root->branch_size : 0)))
    /* Invalid arguments.  */
    abort ();
  {
    gl_listelement_equals_fn equals = list->base.equals_fn;
    /* Iterate across all elements.  */
    gl_list_node_t node = list->root;
    iterstack_t stack;
    iterstack_item_t *stack_ptr = &stack[0];
    size_t index = 0;

    if (start_index == 0)
      {
        /* Consider all elements.  */
        for (;;)
          {
            /* Descend on left branch.  */
            for (;;)
              {
                if (node == NULL)
                  break;
                stack_ptr->node = node;
                stack_ptr->rightp = 0;
                node = node->left;
                stack_ptr++;
              }
            /* Climb up again.  */
            for (;;)
              {
                if (stack_ptr == &stack[0])
                  return NULL;
                stack_ptr--;
                if (!stack_ptr->rightp)
                  break;
              }
            node = stack_ptr->node;
            /* Test against current element.  */
            if (equals != NULL ? equals (elt, node->value) : elt == node->value)
              return node;
            index++;
            if (index >= end_index)
              return NULL;
            /* Descend on right branch.  */
            stack_ptr->rightp = 1;
            node = node->right;
            stack_ptr++;
          }
      }
    else
      {
        /* Consider only elements at indices >= start_index.
           In this case, rightp contains the difference between the start_index
           for the parent node and the one for the child node (0 when the child
           node is the parent's left child, > 0 when the child is the parent's
           right child).  */
        for (;;)
          {
            /* Descend on left branch.  */
            for (;;)
              {
                if (node == NULL)
                  break;
                if (node->branch_size <= start_index)
                  break;
                stack_ptr->node = node;
                stack_ptr->rightp = 0;
                node = node->left;
                stack_ptr++;
              }
            /* Climb up again.  */
            for (;;)
              {
                if (stack_ptr == &stack[0])
                  return NULL;
                stack_ptr--;
                if (!stack_ptr->rightp)
                  break;
                start_index += stack_ptr->rightp;
              }
            node = stack_ptr->node;
            {
              size_t left_branch_size1 =
                (node->left != NULL ? node->left->branch_size : 0) + 1;
              if (start_index < left_branch_size1)
                {
                  /* Test against current element.  */
                  if (equals != NULL ? equals (elt, node->value) : elt == node->value)
                    return node;
                  /* Now that we have considered all indices < left_branch_size1,
                     we can increment start_index.  */
                  start_index = left_branch_size1;
                }
              index++;
              if (index >= end_index)
                return NULL;
              /* Descend on right branch.  */
              start_index -= left_branch_size1;
              stack_ptr->rightp = left_branch_size1;
            }
            node = node->right;
            stack_ptr++;
          }
      }
  }
}

static size_t _GL_ATTRIBUTE_PURE
gl_tree_indexof_from_to (gl_list_t list, size_t start_index, size_t end_index,
                         const void *elt)
{
  if (!(start_index <= end_index
        && end_index <= (list->root != NULL ? list->root->branch_size : 0)))
    /* Invalid arguments.  */
    abort ();
  {
    gl_listelement_equals_fn equals = list->base.equals_fn;
    /* Iterate across all elements.  */
    gl_list_node_t node = list->root;
    iterstack_t stack;
    iterstack_item_t *stack_ptr = &stack[0];
    size_t index = 0;

    if (start_index == 0)
      {
        /* Consider all elements.  */
        for (;;)
          {
            /* Descend on left branch.  */
            for (;;)
              {
                if (node == NULL)
                  break;
                stack_ptr->node = node;
                stack_ptr->rightp = 0;
                node = node->left;
                stack_ptr++;
              }
            /* Climb up again.  */
            for (;;)
              {
                if (stack_ptr == &stack[0])
                  return (size_t)(-1);
                stack_ptr--;
                if (!stack_ptr->rightp)
                  break;
              }
            node = stack_ptr->node;
            /* Test against current element.  */
            if (equals != NULL ? equals (elt, node->value) : elt == node->value)
              return index;
            index++;
            if (index >= end_index)
              return (size_t)(-1);
            /* Descend on right branch.  */
            stack_ptr->rightp = 1;
            node = node->right;
            stack_ptr++;
          }
      }
    else
      {
        /* Consider only elements at indices >= start_index.
           In this case, rightp contains the difference between the start_index
           for the parent node and the one for the child node (0 when the child
           node is the parent's left child, > 0 when the child is the parent's
           right child).  */
        for (;;)
          {
            /* Descend on left branch.  */
            for (;;)
              {
                if (node == NULL)
                  break;
                if (node->branch_size <= start_index)
                  break;
                stack_ptr->node = node;
                stack_ptr->rightp = 0;
                node = node->left;
                stack_ptr++;
              }
            /* Climb up again.  */
            for (;;)
              {
                if (stack_ptr == &stack[0])
                  return (size_t)(-1);
                stack_ptr--;
                if (!stack_ptr->rightp)
                  break;
                start_index += stack_ptr->rightp;
              }
            node = stack_ptr->node;
            {
              size_t left_branch_size1 =
                (node->left != NULL ? node->left->branch_size : 0) + 1;
              if (start_index < left_branch_size1)
                {
                  /* Test against current element.  */
                  if (equals != NULL ? equals (elt, node->value) : elt == node->value)
                    return index;
                  /* Now that we have considered all indices < left_branch_size1,
                     we can increment start_index.  */
                  start_index = left_branch_size1;
                }
              index++;
              if (index >= end_index)
                return (size_t)(-1);
              /* Descend on right branch.  */
              start_index -= left_branch_size1;
              stack_ptr->rightp = left_branch_size1;
            }
            node = node->right;
            stack_ptr++;
          }
      }
  }
}

#endif

static gl_list_node_t
gl_tree_nx_add_at (gl_list_t list, size_t position, const void *elt)
{
  size_t count = (list->root != NULL ? list->root->branch_size : 0);

  if (!(position <= count))
    /* Invalid argument.  */
    abort ();
  if (position == count)
    return gl_tree_nx_add_last (list, elt);
  else
    return gl_tree_nx_add_before (list, node_at (list->root, position), elt);
}

static bool
gl_tree_remove_node (gl_list_t list, gl_list_node_t node)
{
#if WITH_HASHTABLE
  /* Remove node from the hash table.
     Note that this is only possible _before_ the node is removed from the
     tree structure, because remove_from_bucket() uses node_position().  */
  remove_from_bucket (list, node);
#endif

  gl_tree_remove_node_from_tree (list, node);

  if (list->base.dispose_fn != NULL)
    list->base.dispose_fn (node->value);
  free (node);
  return true;
}

static bool
gl_tree_remove_at (gl_list_t list, size_t position)
{
  gl_list_node_t node = list->root;

  if (!(node != NULL && position < node->branch_size))
    /* Invalid argument.  */
    abort ();
  node = node_at (node, position);
  return gl_tree_remove_node (list, node);
}

static bool
gl_tree_remove (gl_list_t list, const void *elt)
{
  if (list->root != NULL)
    {
      gl_list_node_t node =
        gl_tree_search_from_to (list, 0, list->root->branch_size, elt);

      if (node != NULL)
        return gl_tree_remove_node (list, node);
    }
  return false;
}

#if !WITH_HASHTABLE

static void
gl_tree_list_free (gl_list_t list)
{
  /* Iterate across all elements in post-order.  */
  gl_list_node_t node = list->root;
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
          if (list->base.dispose_fn != NULL)
            list->base.dispose_fn (node->value);
          free (node);
        }
      /* Descend on right branch.  */
      stack_ptr->rightp = true;
      node = node->right;
      stack_ptr++;
    }
 done_iterate:
  free (list);
}

#endif

/* --------------------- gl_list_iterator_t Data Type --------------------- */

static gl_list_iterator_t _GL_ATTRIBUTE_PURE
gl_tree_iterator (gl_list_t list)
{
  gl_list_iterator_t result;
  gl_list_node_t node;

  result.vtable = list->base.vtable;
  result.list = list;
  /* Start node is the leftmost node.  */
  node = list->root;
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

static gl_list_iterator_t _GL_ATTRIBUTE_PURE
gl_tree_iterator_from_to (gl_list_t list, size_t start_index, size_t end_index)
{
  size_t count = (list->root != NULL ? list->root->branch_size : 0);
  gl_list_iterator_t result;

  if (!(start_index <= end_index && end_index <= count))
    /* Invalid arguments.  */
    abort ();
  result.vtable = list->base.vtable;
  result.list = list;
  /* Start node is the node at position start_index.  */
  result.p = (start_index < count ? node_at (list->root, start_index) : NULL);
  /* End point is the node at position end_index.  */
  result.q = (end_index < count ? node_at (list->root, end_index) : NULL);
#if defined GCC_LINT || defined lint
  result.i = 0;
  result.j = 0;
  result.count = 0;
#endif

  return result;
}

static bool
gl_tree_iterator_next (gl_list_iterator_t *iterator,
                       const void **eltp, gl_list_node_t *nodep)
{
  if (iterator->p != iterator->q)
    {
      gl_list_node_t node = (gl_list_node_t) iterator->p;
      *eltp = node->value;
      if (nodep != NULL)
        *nodep = node;
      /* Advance to the next node.  */
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
      iterator->p = node;
      return true;
    }
  else
    return false;
}

static void
gl_tree_iterator_free (gl_list_iterator_t *iterator  _GL_ATTRIBUTE_MAYBE_UNUSED)
{
}

/* ---------------------- Sorted gl_list_t Data Type ---------------------- */

static gl_list_node_t _GL_ATTRIBUTE_PURE
gl_tree_sortedlist_search (gl_list_t list, gl_listelement_compar_fn compar,
                           const void *elt)
{
  gl_list_node_t node;

  for (node = list->root; node != NULL; )
    {
      int cmp = compar (node->value, elt);

      if (cmp < 0)
        node = node->right;
      else if (cmp > 0)
        node = node->left;
      else /* cmp == 0 */
        {
          /* We have an element equal to ELT.  But we need the leftmost such
             element.  */
          gl_list_node_t found = node;
          node = node->left;
          for (; node != NULL; )
            {
              int cmp2 = compar (node->value, elt);

              if (cmp2 < 0)
                node = node->right;
              else if (cmp2 > 0)
                /* The list was not sorted.  */
                abort ();
              else /* cmp2 == 0 */
                {
                  found = node;
                  node = node->left;
                }
            }
          return found;
        }
    }
  return NULL;
}

static gl_list_node_t _GL_ATTRIBUTE_PURE
gl_tree_sortedlist_search_from_to (gl_list_t list,
                                   gl_listelement_compar_fn compar,
                                   size_t low, size_t high,
                                   const void *elt)
{
  gl_list_node_t node;

  if (!(low <= high
        && high <= (list->root != NULL ? list->root->branch_size : 0)))
    /* Invalid arguments.  */
    abort ();

  for (node = list->root; node != NULL; )
    {
      size_t left_branch_size =
        (node->left != NULL ? node->left->branch_size : 0);

      if (low > left_branch_size)
        {
          low -= left_branch_size + 1;
          high -= left_branch_size + 1;
          node = node->right;
        }
      else if (high <= left_branch_size)
        node = node->left;
      else
        {
          /* Here low <= left_branch_size < high.  */
          int cmp = compar (node->value, elt);

          if (cmp < 0)
            {
              low = 0;
              high -= left_branch_size + 1;
              node = node->right;
            }
          else if (cmp > 0)
            node = node->left;
          else /* cmp == 0 */
            {
              /* We have an element equal to ELT.  But we need the leftmost
                 such element.  */
              gl_list_node_t found = node;
              node = node->left;
              for (; node != NULL; )
                {
                  size_t left_branch_size2 =
                    (node->left != NULL ? node->left->branch_size : 0);

                  if (low > left_branch_size2)
                    {
                      low -= left_branch_size2 + 1;
                      node = node->right;
                    }
                  else
                    {
                      /* Here low <= left_branch_size2.  */
                      int cmp2 = compar (node->value, elt);

                      if (cmp2 < 0)
                        {
                          low = 0;
                          node = node->right;
                        }
                      else if (cmp2 > 0)
                        /* The list was not sorted.  */
                        abort ();
                      else /* cmp2 == 0 */
                        {
                          found = node;
                          node = node->left;
                        }
                    }
                }
              return found;
            }
        }
    }
  return NULL;
}

static size_t _GL_ATTRIBUTE_PURE
gl_tree_sortedlist_indexof (gl_list_t list, gl_listelement_compar_fn compar,
                            const void *elt)
{
  gl_list_node_t node;
  size_t position;

  for (node = list->root, position = 0; node != NULL; )
    {
      int cmp = compar (node->value, elt);

      if (cmp < 0)
        {
          if (node->left != NULL)
            position += node->left->branch_size;
          position++;
          node = node->right;
        }
      else if (cmp > 0)
        node = node->left;
      else /* cmp == 0 */
        {
          /* We have an element equal to ELT.  But we need the leftmost such
             element.  */
          size_t found_position =
            position + (node->left != NULL ? node->left->branch_size : 0);
          node = node->left;
          for (; node != NULL; )
            {
              int cmp2 = compar (node->value, elt);

              if (cmp2 < 0)
                {
                  if (node->left != NULL)
                    position += node->left->branch_size;
                  position++;
                  node = node->right;
                }
              else if (cmp2 > 0)
                /* The list was not sorted.  */
                abort ();
              else /* cmp2 == 0 */
                {
                  found_position =
                    position
                    + (node->left != NULL ? node->left->branch_size : 0);
                  node = node->left;
                }
            }
          return found_position;
        }
    }
  return (size_t)(-1);
}

static size_t _GL_ATTRIBUTE_PURE
gl_tree_sortedlist_indexof_from_to (gl_list_t list,
                                    gl_listelement_compar_fn compar,
                                    size_t low, size_t high,
                                    const void *elt)
{
  gl_list_node_t node;
  size_t position;

  if (!(low <= high
        && high <= (list->root != NULL ? list->root->branch_size : 0)))
    /* Invalid arguments.  */
    abort ();

  for (node = list->root, position = 0; node != NULL; )
    {
      size_t left_branch_size =
        (node->left != NULL ? node->left->branch_size : 0);

      if (low > left_branch_size)
        {
          low -= left_branch_size + 1;
          high -= left_branch_size + 1;
          position += left_branch_size + 1;
          node = node->right;
        }
      else if (high <= left_branch_size)
        node = node->left;
      else
        {
          /* Here low <= left_branch_size < high.  */
          int cmp = compar (node->value, elt);

          if (cmp < 0)
            {
              low = 0;
              high -= left_branch_size + 1;
              position += left_branch_size + 1;
              node = node->right;
            }
          else if (cmp > 0)
            node = node->left;
          else /* cmp == 0 */
            {
              /* We have an element equal to ELT.  But we need the leftmost
                 such element.  */
              size_t found_position =
                position + (node->left != NULL ? node->left->branch_size : 0);
              node = node->left;
              for (; node != NULL; )
                {
                  size_t left_branch_size2 =
                    (node->left != NULL ? node->left->branch_size : 0);

                  if (low > left_branch_size2)
                    {
                      low -= left_branch_size2 + 1;
                      node = node->right;
                    }
                  else
                    {
                      /* Here low <= left_branch_size2.  */
                      int cmp2 = compar (node->value, elt);

                      if (cmp2 < 0)
                        {
                          position += left_branch_size2 + 1;
                          node = node->right;
                        }
                      else if (cmp2 > 0)
                        /* The list was not sorted.  */
                        abort ();
                      else /* cmp2 == 0 */
                        {
                          found_position = position + left_branch_size2;
                          node = node->left;
                        }
                    }
                }
              return found_position;
            }
        }
    }
  return (size_t)(-1);
}

static gl_list_node_t
gl_tree_sortedlist_nx_add (gl_list_t list, gl_listelement_compar_fn compar,
                           const void *elt)
{
  gl_list_node_t node = list->root;

  if (node == NULL)
    return gl_tree_nx_add_first (list, elt);

  for (;;)
    {
      int cmp = compar (node->value, elt);

      if (cmp < 0)
        {
          if (node->right == NULL)
            return gl_tree_nx_add_after (list, node, elt);
          node = node->right;
        }
      else if (cmp > 0)
        {
          if (node->left == NULL)
            return gl_tree_nx_add_before (list, node, elt);
          node = node->left;
        }
      else /* cmp == 0 */
        return gl_tree_nx_add_before (list, node, elt);
    }
}

static bool
gl_tree_sortedlist_remove (gl_list_t list, gl_listelement_compar_fn compar,
                           const void *elt)
{
  gl_list_node_t node = gl_tree_sortedlist_search (list, compar, elt);
  if (node != NULL)
    return gl_tree_remove_node (list, node);
  else
    return false;
}
