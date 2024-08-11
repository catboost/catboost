/* Sequential list data type implemented by a hash table with a binary tree.
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

/* Common code of gl_avltreehash_list.c and gl_rbtreehash_list.c.  */

static gl_list_node_t
gl_tree_search_from_to (gl_list_t list, size_t start_index, size_t end_index,
                        const void *elt)
{
  if (!(start_index <= end_index
        && end_index <= (list->root != NULL ? list->root->branch_size : 0)))
    /* Invalid arguments.  */
    abort ();
  {
    size_t hashcode =
      (list->base.hashcode_fn != NULL
       ? list->base.hashcode_fn (elt)
       : (size_t)(uintptr_t) elt);
    size_t bucket = hashcode % list->table_size;
    gl_listelement_equals_fn equals = list->base.equals_fn;
    gl_hash_entry_t entry;

    if (list->base.allow_duplicates)
      {
        for (entry = list->table[bucket]; entry != NULL; entry = entry->hash_next)
          if (entry->hashcode == hashcode)
            {
              if (((struct gl_multiple_nodes *) entry)->magic == MULTIPLE_NODES_MAGIC)
                {
                  /* An entry representing multiple nodes.  */
                  gl_oset_t nodes = ((struct gl_multiple_nodes *) entry)->nodes;
                  /* The first node is interesting.  */
                  gl_list_node_t node = gl_oset_first (nodes);
                  if (equals != NULL ? equals (elt, node->value) : elt == node->value)
                    {
                      /* All nodes in the entry are equal to the given ELT.  */
                      if (start_index == 0)
                        {
                          /* We have to return only the one at the minimal
                             position, and this is the first one in the ordered
                             set.  */
                          if (end_index == list->root->branch_size
                              || node_position (node) < end_index)
                            return node;
                        }
                      else
                        {
                          /* We have to return only the one at the minimal
                             position >= start_index.  */
                          const void *nodes_elt;
                          if (gl_oset_search_atleast (nodes,
                                                      compare_position_threshold,
                                                      (void *)(uintptr_t)start_index,
                                                      &nodes_elt))
                            {
                              node = (gl_list_node_t) nodes_elt;
                              if (end_index == list->root->branch_size
                                  || node_position (node) < end_index)
                                return node;
                            }
                        }
                      break;
                    }
                }
              else
                {
                  /* An entry representing a single node.  */
                  gl_list_node_t node = (struct gl_list_node_impl *) entry;
                  if (equals != NULL ? equals (elt, node->value) : elt == node->value)
                    {
                      bool position_in_bounds;
                      if (start_index == 0 && end_index == list->root->branch_size)
                        position_in_bounds = true;
                      else
                        {
                          size_t position = node_position (node);
                          position_in_bounds =
                            (position >= start_index && position < end_index);
                        }
                      if (position_in_bounds)
                        return node;
                      break;
                    }
                }
            }
      }
    else
      {
        /* If no duplicates are allowed, multiple nodes are not needed.  */
        for (entry = list->table[bucket]; entry != NULL; entry = entry->hash_next)
          if (entry->hashcode == hashcode)
            {
              gl_list_node_t node = (struct gl_list_node_impl *) entry;
              if (equals != NULL ? equals (elt, node->value) : elt == node->value)
                {
                  bool position_in_bounds;
                  if (start_index == 0 && end_index == list->root->branch_size)
                    position_in_bounds = true;
                  else
                    {
                      size_t position = node_position (node);
                      position_in_bounds =
                        (position >= start_index && position < end_index);
                    }
                  if (position_in_bounds)
                    return node;
                  break;
                }
            }
      }

    return NULL;
  }
}

static size_t
gl_tree_indexof_from_to (gl_list_t list, size_t start_index, size_t end_index,
                         const void *elt)
{
  gl_list_node_t node =
    gl_tree_search_from_to (list, start_index, end_index, elt);

  if (node != NULL)
    return node_position (node);
  else
    return (size_t)(-1);
}

static void
gl_tree_list_free (gl_list_t list)
{
  if (list->base.allow_duplicates)
    {
      /* Free the ordered sets in the hash buckets.  */
      size_t i;

      for (i = list->table_size; i > 0; )
        {
          gl_hash_entry_t entry = list->table[--i];

          while (entry != NULL)
            {
              gl_hash_entry_t next = entry->hash_next;

              if (((struct gl_multiple_nodes *) entry)->magic == MULTIPLE_NODES_MAGIC)
                {
                  gl_oset_t nodes = ((struct gl_multiple_nodes *) entry)->nodes;

                  gl_oset_free (nodes);
                  free (entry);
                }

              entry = next;
            }
        }
    }

  /* Iterate across all elements in post-order.  */
  {
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
  }
 done_iterate:
  free (list->table);
  free (list);
}
