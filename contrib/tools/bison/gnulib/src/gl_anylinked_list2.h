/* Sequential list data type implemented by a linked list.
   Copyright (C) 2006-2013 Free Software Foundation, Inc.
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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Common code of gl_linked_list.c and gl_linkedhash_list.c.  */

/* If the symbol SIGNAL_SAFE_LIST is defined, the code is compiled in such
   a way that a gl_list_t data structure may be used from within a signal
   handler.  The operations allowed in the signal handler are:
     gl_list_iterator, gl_list_iterator_next, gl_list_iterator_free.
   The list and node fields that are therefore accessed from the signal handler
   are:
     list->root, node->next, node->value.
   We are careful to make modifications to these fields only in an order
   that maintains the consistency of the list data structure at any moment,
   and we use 'volatile' assignments to prevent the compiler from reordering
   such assignments.  */
#ifdef SIGNAL_SAFE_LIST
# define ASYNCSAFE(type) *(volatile type *)&
#else
# define ASYNCSAFE(type)
#endif

/* -------------------------- gl_list_t Data Type -------------------------- */

static gl_list_t
gl_linked_nx_create_empty (gl_list_implementation_t implementation,
                           gl_listelement_equals_fn equals_fn,
                           gl_listelement_hashcode_fn hashcode_fn,
                           gl_listelement_dispose_fn dispose_fn,
                           bool allow_duplicates)
{
  struct gl_list_impl *list =
    (struct gl_list_impl *) malloc (sizeof (struct gl_list_impl));

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
  list->root.next = &list->root;
  list->root.prev = &list->root;
  list->count = 0;

  return list;

#if WITH_HASHTABLE
 fail:
  free (list);
  return NULL;
#endif
}

static gl_list_t
gl_linked_nx_create (gl_list_implementation_t implementation,
                  gl_listelement_equals_fn equals_fn,
                  gl_listelement_hashcode_fn hashcode_fn,
                  gl_listelement_dispose_fn dispose_fn,
                  bool allow_duplicates,
                  size_t count, const void **contents)
{
  struct gl_list_impl *list =
    (struct gl_list_impl *) malloc (sizeof (struct gl_list_impl));
  gl_list_node_t tail;

  if (list == NULL)
    return NULL;

  list->base.vtable = implementation;
  list->base.equals_fn = equals_fn;
  list->base.hashcode_fn = hashcode_fn;
  list->base.dispose_fn = dispose_fn;
  list->base.allow_duplicates = allow_duplicates;
#if WITH_HASHTABLE
  {
    size_t estimate = xsum (count, count / 2); /* 1.5 * count */
    if (estimate < 10)
      estimate = 10;
    list->table_size = next_prime (estimate);
    if (size_overflow_p (xtimes (list->table_size, sizeof (gl_hash_entry_t))))
      goto fail1;
    list->table =
      (gl_hash_entry_t *) calloc (list->table_size, sizeof (gl_hash_entry_t));
    if (list->table == NULL)
      goto fail1;
  }
#endif
  list->count = count;
  tail = &list->root;
  for (; count > 0; contents++, count--)
    {
      gl_list_node_t node =
        (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

      if (node == NULL)
        goto fail2;

      node->value = *contents;
#if WITH_HASHTABLE
      node->h.hashcode =
        (list->base.hashcode_fn != NULL
         ? list->base.hashcode_fn (node->value)
         : (size_t)(uintptr_t) node->value);

      /* Add node to the hash table.  */
      if (add_to_bucket (list, node) < 0)
        {
          free (node);
          goto fail2;
        }
#endif

      /* Add node to the list.  */
      node->prev = tail;
      tail->next = node;
      tail = node;
    }
  tail->next = &list->root;
  list->root.prev = tail;

  return list;

 fail2:
  {
    gl_list_node_t node;

    for (node = tail; node != &list->root; )
      {
        gl_list_node_t prev = node->prev;

        free (node);
        node = prev;
      }
  }
#if WITH_HASHTABLE
  free (list->table);
 fail1:
#endif
  free (list);
  return NULL;
}

static size_t
gl_linked_size (gl_list_t list)
{
  return list->count;
}

static const void *
gl_linked_node_value (gl_list_t list, gl_list_node_t node)
{
  return node->value;
}

static int
gl_linked_node_nx_set_value (gl_list_t list, gl_list_node_t node,
                             const void *elt)
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
              gl_list_node_t before_removed = node->prev;
              gl_list_node_t after_removed = node->next;
              ASYNCSAFE(gl_list_node_t) before_removed->next = after_removed;
              after_removed->prev = before_removed;
              list->count--;
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

static gl_list_node_t
gl_linked_next_node (gl_list_t list, gl_list_node_t node)
{
  return (node->next != &list->root ? node->next : NULL);
}

static gl_list_node_t
gl_linked_previous_node (gl_list_t list, gl_list_node_t node)
{
  return (node->prev != &list->root ? node->prev : NULL);
}

static const void *
gl_linked_get_at (gl_list_t list, size_t position)
{
  size_t count = list->count;
  gl_list_node_t node;

  if (!(position < count))
    /* Invalid argument.  */
    abort ();
  /* Here we know count > 0.  */
  if (position <= ((count - 1) / 2))
    {
      node = list->root.next;
      for (; position > 0; position--)
        node = node->next;
    }
  else
    {
      position = count - 1 - position;
      node = list->root.prev;
      for (; position > 0; position--)
        node = node->prev;
    }
  return node->value;
}

static gl_list_node_t
gl_linked_nx_set_at (gl_list_t list, size_t position, const void *elt)
{
  size_t count = list->count;
  gl_list_node_t node;

  if (!(position < count))
    /* Invalid argument.  */
    abort ();
  /* Here we know count > 0.  */
  if (position <= ((count - 1) / 2))
    {
      node = list->root.next;
      for (; position > 0; position--)
        node = node->next;
    }
  else
    {
      position = count - 1 - position;
      node = list->root.prev;
      for (; position > 0; position--)
        node = node->prev;
    }
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
              gl_list_node_t before_removed = node->prev;
              gl_list_node_t after_removed = node->next;
              ASYNCSAFE(gl_list_node_t) before_removed->next = after_removed;
              after_removed->prev = before_removed;
              list->count--;
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

static gl_list_node_t
gl_linked_search_from_to (gl_list_t list, size_t start_index, size_t end_index,
                          const void *elt)
{
  size_t count = list->count;

  if (!(start_index <= end_index && end_index <= count))
    /* Invalid arguments.  */
    abort ();
  {
#if WITH_HASHTABLE
    size_t hashcode =
      (list->base.hashcode_fn != NULL
       ? list->base.hashcode_fn (elt)
       : (size_t)(uintptr_t) elt);
    size_t bucket = hashcode % list->table_size;
    gl_listelement_equals_fn equals = list->base.equals_fn;

    if (!list->base.allow_duplicates)
      {
        /* Look for the first match in the hash bucket.  */
        gl_list_node_t found = NULL;
        gl_list_node_t node;

        for (node = (gl_list_node_t) list->table[bucket];
             node != NULL;
             node = (gl_list_node_t) node->h.hash_next)
          if (node->h.hashcode == hashcode
              && (equals != NULL
                  ? equals (elt, node->value)
                  : elt == node->value))
            {
              found = node;
              break;
            }
        if (start_index > 0)
          /* Look whether found's index is < start_index.  */
          for (node = list->root.next; ; node = node->next)
            {
              if (node == found)
                return NULL;
              if (--start_index == 0)
                break;
            }
        if (end_index < count)
          /* Look whether found's index is >= end_index.  */
          {
            end_index = count - end_index;
            for (node = list->root.prev; ; node = node->prev)
              {
                if (node == found)
                  return NULL;
                if (--end_index == 0)
                  break;
              }
          }
        return found;
      }
    else
      {
        /* Look whether there is more than one match in the hash bucket.  */
        bool multiple_matches = false;
        gl_list_node_t first_match = NULL;
        gl_list_node_t node;

        for (node = (gl_list_node_t) list->table[bucket];
             node != NULL;
             node = (gl_list_node_t) node->h.hash_next)
          if (node->h.hashcode == hashcode
              && (equals != NULL
                  ? equals (elt, node->value)
                  : elt == node->value))
            {
              if (first_match == NULL)
                first_match = node;
              else
                {
                  multiple_matches = true;
                  break;
                }
            }
        if (multiple_matches)
          {
            /* We need the match with the smallest index.  But we don't have
               a fast mapping node -> index.  So we have to walk the list.  */
            end_index -= start_index;
            node = list->root.next;
            for (; start_index > 0; start_index--)
              node = node->next;

            for (;
                 end_index > 0;
                 node = node->next, end_index--)
              if (node->h.hashcode == hashcode
                  && (equals != NULL
                      ? equals (elt, node->value)
                      : elt == node->value))
                return node;
            /* The matches must have all been at indices < start_index or
               >= end_index.  */
            return NULL;
          }
        else
          {
            if (start_index > 0)
              /* Look whether first_match's index is < start_index.  */
              for (node = list->root.next; node != &list->root; node = node->next)
                {
                  if (node == first_match)
                    return NULL;
                  if (--start_index == 0)
                    break;
                }
            if (end_index < list->count)
              /* Look whether first_match's index is >= end_index.  */
              {
                end_index = list->count - end_index;
                for (node = list->root.prev; ; node = node->prev)
                  {
                    if (node == first_match)
                      return NULL;
                    if (--end_index == 0)
                      break;
                  }
              }
            return first_match;
          }
      }
#else
    gl_listelement_equals_fn equals = list->base.equals_fn;
    gl_list_node_t node = list->root.next;

    end_index -= start_index;
    for (; start_index > 0; start_index--)
      node = node->next;

    if (equals != NULL)
      {
        for (; end_index > 0; node = node->next, end_index--)
          if (equals (elt, node->value))
            return node;
      }
    else
      {
        for (; end_index > 0; node = node->next, end_index--)
          if (elt == node->value)
            return node;
      }
    return NULL;
#endif
  }
}

static size_t
gl_linked_indexof_from_to (gl_list_t list, size_t start_index, size_t end_index,
                           const void *elt)
{
  size_t count = list->count;

  if (!(start_index <= end_index && end_index <= count))
    /* Invalid arguments.  */
    abort ();
  {
#if WITH_HASHTABLE
    /* Here the hash table doesn't help much.  It only allows us to minimize
       the number of equals() calls, by looking up first the node and then
       its index.  */
    size_t hashcode =
      (list->base.hashcode_fn != NULL
       ? list->base.hashcode_fn (elt)
       : (size_t)(uintptr_t) elt);
    size_t bucket = hashcode % list->table_size;
    gl_listelement_equals_fn equals = list->base.equals_fn;
    gl_list_node_t node;

    /* First step: Look up the node.  */
    if (!list->base.allow_duplicates)
      {
        /* Look for the first match in the hash bucket.  */
        for (node = (gl_list_node_t) list->table[bucket];
             node != NULL;
             node = (gl_list_node_t) node->h.hash_next)
          if (node->h.hashcode == hashcode
              && (equals != NULL
                  ? equals (elt, node->value)
                  : elt == node->value))
            break;
      }
    else
      {
        /* Look whether there is more than one match in the hash bucket.  */
        bool multiple_matches = false;
        gl_list_node_t first_match = NULL;

        for (node = (gl_list_node_t) list->table[bucket];
             node != NULL;
             node = (gl_list_node_t) node->h.hash_next)
          if (node->h.hashcode == hashcode
              && (equals != NULL
                  ? equals (elt, node->value)
                  : elt == node->value))
            {
              if (first_match == NULL)
                first_match = node;
              else
                {
                  multiple_matches = true;
                  break;
                }
            }
        if (multiple_matches)
          {
            /* We need the match with the smallest index.  But we don't have
               a fast mapping node -> index.  So we have to walk the list.  */
            size_t index;

            index = start_index;
            node = list->root.next;
            for (; start_index > 0; start_index--)
              node = node->next;

            for (;
                 index < end_index;
                 node = node->next, index++)
              if (node->h.hashcode == hashcode
                  && (equals != NULL
                      ? equals (elt, node->value)
                      : elt == node->value))
                return index;
            /* The matches must have all been at indices < start_index or
               >= end_index.  */
            return (size_t)(-1);
          }
        node = first_match;
      }

    /* Second step: Look up the index of the node.  */
    if (node == NULL)
      return (size_t)(-1);
    else
      {
        size_t index = 0;

        for (; node->prev != &list->root; node = node->prev)
          index++;

        if (index >= start_index && index < end_index)
          return index;
        else
          return (size_t)(-1);
      }
#else
    gl_listelement_equals_fn equals = list->base.equals_fn;
    size_t index = start_index;
    gl_list_node_t node = list->root.next;

    for (; start_index > 0; start_index--)
      node = node->next;

    if (equals != NULL)
      {
        for (;
             index < end_index;
             node = node->next, index++)
          if (equals (elt, node->value))
            return index;
      }
    else
      {
        for (;
             index < end_index;
             node = node->next, index++)
          if (elt == node->value)
            return index;
      }
    return (size_t)(-1);
#endif
  }
}

static gl_list_node_t
gl_linked_nx_add_first (gl_list_t list, const void *elt)
{
  gl_list_node_t node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (node == NULL)
    return NULL;

  ASYNCSAFE(const void *) node->value = elt;
#if WITH_HASHTABLE
  node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (node->value)
     : (size_t)(uintptr_t) node->value);

  /* Add node to the hash table.  */
  if (add_to_bucket (list, node) < 0)
    {
      free (node);
      return NULL;
    }
#endif

  /* Add node to the list.  */
  node->prev = &list->root;
  ASYNCSAFE(gl_list_node_t) node->next = list->root.next;
  node->next->prev = node;
  ASYNCSAFE(gl_list_node_t) list->root.next = node;
  list->count++;

#if WITH_HASHTABLE
  hash_resize_after_add (list);
#endif

  return node;
}

static gl_list_node_t
gl_linked_nx_add_last (gl_list_t list, const void *elt)
{
  gl_list_node_t node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (node == NULL)
    return NULL;

  ASYNCSAFE(const void *) node->value = elt;
#if WITH_HASHTABLE
  node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (node->value)
     : (size_t)(uintptr_t) node->value);

  /* Add node to the hash table.  */
  if (add_to_bucket (list, node) < 0)
    {
      free (node);
      return NULL;
    }
#endif

  /* Add node to the list.  */
  ASYNCSAFE(gl_list_node_t) node->next = &list->root;
  node->prev = list->root.prev;
  ASYNCSAFE(gl_list_node_t) node->prev->next = node;
  list->root.prev = node;
  list->count++;

#if WITH_HASHTABLE
  hash_resize_after_add (list);
#endif

  return node;
}

static gl_list_node_t
gl_linked_nx_add_before (gl_list_t list, gl_list_node_t node, const void *elt)
{
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  ASYNCSAFE(const void *) new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);

  /* Add new_node to the hash table.  */
  if (add_to_bucket (list, new_node) < 0)
    {
      free (new_node);
      return NULL;
    }
#endif

  /* Add new_node to the list.  */
  ASYNCSAFE(gl_list_node_t) new_node->next = node;
  new_node->prev = node->prev;
  ASYNCSAFE(gl_list_node_t) new_node->prev->next = new_node;
  node->prev = new_node;
  list->count++;

#if WITH_HASHTABLE
  hash_resize_after_add (list);
#endif

  return new_node;
}

static gl_list_node_t
gl_linked_nx_add_after (gl_list_t list, gl_list_node_t node, const void *elt)
{
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  ASYNCSAFE(const void *) new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);

  /* Add new_node to the hash table.  */
  if (add_to_bucket (list, new_node) < 0)
    {
      free (new_node);
      return NULL;
    }
#endif

  /* Add new_node to the list.  */
  new_node->prev = node;
  ASYNCSAFE(gl_list_node_t) new_node->next = node->next;
  new_node->next->prev = new_node;
  ASYNCSAFE(gl_list_node_t) node->next = new_node;
  list->count++;

#if WITH_HASHTABLE
  hash_resize_after_add (list);
#endif

  return new_node;
}

static gl_list_node_t
gl_linked_nx_add_at (gl_list_t list, size_t position, const void *elt)
{
  size_t count = list->count;
  gl_list_node_t new_node;

  if (!(position <= count))
    /* Invalid argument.  */
    abort ();

  new_node = (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));
  if (new_node == NULL)
    return NULL;

  ASYNCSAFE(const void *) new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);

  /* Add new_node to the hash table.  */
  if (add_to_bucket (list, new_node) < 0)
    {
      free (new_node);
      return NULL;
    }
#endif

  /* Add new_node to the list.  */
  if (position <= (count / 2))
    {
      gl_list_node_t node;

      node = &list->root;
      for (; position > 0; position--)
        node = node->next;
      new_node->prev = node;
      ASYNCSAFE(gl_list_node_t) new_node->next = node->next;
      new_node->next->prev = new_node;
      ASYNCSAFE(gl_list_node_t) node->next = new_node;
    }
  else
    {
      gl_list_node_t node;

      position = count - position;
      node = &list->root;
      for (; position > 0; position--)
        node = node->prev;
      ASYNCSAFE(gl_list_node_t) new_node->next = node;
      new_node->prev = node->prev;
      ASYNCSAFE(gl_list_node_t) new_node->prev->next = new_node;
      node->prev = new_node;
    }
  list->count++;

#if WITH_HASHTABLE
  hash_resize_after_add (list);
#endif

  return new_node;
}

static bool
gl_linked_remove_node (gl_list_t list, gl_list_node_t node)
{
  gl_list_node_t prev;
  gl_list_node_t next;

#if WITH_HASHTABLE
  /* Remove node from the hash table.  */
  remove_from_bucket (list, node);
#endif

  /* Remove node from the list.  */
  prev = node->prev;
  next = node->next;

  ASYNCSAFE(gl_list_node_t) prev->next = next;
  next->prev = prev;
  list->count--;

  if (list->base.dispose_fn != NULL)
    list->base.dispose_fn (node->value);
  free (node);
  return true;
}

static bool
gl_linked_remove_at (gl_list_t list, size_t position)
{
  size_t count = list->count;
  gl_list_node_t removed_node;

  if (!(position < count))
    /* Invalid argument.  */
    abort ();
  /* Here we know count > 0.  */
  if (position <= ((count - 1) / 2))
    {
      gl_list_node_t node;
      gl_list_node_t after_removed;

      node = &list->root;
      for (; position > 0; position--)
        node = node->next;
      removed_node = node->next;
      after_removed = node->next->next;
      ASYNCSAFE(gl_list_node_t) node->next = after_removed;
      after_removed->prev = node;
    }
  else
    {
      gl_list_node_t node;
      gl_list_node_t before_removed;

      position = count - 1 - position;
      node = &list->root;
      for (; position > 0; position--)
        node = node->prev;
      removed_node = node->prev;
      before_removed = node->prev->prev;
      node->prev = before_removed;
      ASYNCSAFE(gl_list_node_t) before_removed->next = node;
    }
#if WITH_HASHTABLE
  remove_from_bucket (list, removed_node);
#endif
  list->count--;

  if (list->base.dispose_fn != NULL)
    list->base.dispose_fn (removed_node->value);
  free (removed_node);
  return true;
}

static bool
gl_linked_remove (gl_list_t list, const void *elt)
{
  gl_list_node_t node = gl_linked_search_from_to (list, 0, list->count, elt);

  if (node != NULL)
    return gl_linked_remove_node (list, node);
  else
    return false;
}

static void
gl_linked_list_free (gl_list_t list)
{
  gl_listelement_dispose_fn dispose = list->base.dispose_fn;
  gl_list_node_t node;

  for (node = list->root.next; node != &list->root; )
    {
      gl_list_node_t next = node->next;
      if (dispose != NULL)
        dispose (node->value);
      free (node);
      node = next;
    }
#if WITH_HASHTABLE
  free (list->table);
#endif
  free (list);
}

/* --------------------- gl_list_iterator_t Data Type --------------------- */

static gl_list_iterator_t
gl_linked_iterator (gl_list_t list)
{
  gl_list_iterator_t result;

  result.vtable = list->base.vtable;
  result.list = list;
  result.p = list->root.next;
  result.q = &list->root;
#ifdef lint
  result.i = 0;
  result.j = 0;
  result.count = 0;
#endif

  return result;
}

static gl_list_iterator_t
gl_linked_iterator_from_to (gl_list_t list,
                            size_t start_index, size_t end_index)
{
  gl_list_iterator_t result;
  size_t n1, n2, n3;

  if (!(start_index <= end_index && end_index <= list->count))
    /* Invalid arguments.  */
    abort ();
  result.vtable = list->base.vtable;
  result.list = list;
  n1 = start_index;
  n2 = end_index - start_index;
  n3 = list->count - end_index;
  /* Find the maximum among n1, n2, n3, so as to reduce the number of
     loop iterations to n1 + n2 + n3 - max(n1,n2,n3).  */
  if (n1 > n2 && n1 > n3)
    {
      /* n1 is the maximum, use n2 and n3.  */
      gl_list_node_t node;
      size_t i;

      node = &list->root;
      for (i = n3; i > 0; i--)
        node = node->prev;
      result.q = node;
      for (i = n2; i > 0; i--)
        node = node->prev;
      result.p = node;
    }
  else if (n2 > n3)
    {
      /* n2 is the maximum, use n1 and n3.  */
      gl_list_node_t node;
      size_t i;

      node = list->root.next;
      for (i = n1; i > 0; i--)
        node = node->next;
      result.p = node;

      node = &list->root;
      for (i = n3; i > 0; i--)
        node = node->prev;
      result.q = node;
    }
  else
    {
      /* n3 is the maximum, use n1 and n2.  */
      gl_list_node_t node;
      size_t i;

      node = list->root.next;
      for (i = n1; i > 0; i--)
        node = node->next;
      result.p = node;
      for (i = n2; i > 0; i--)
        node = node->next;
      result.q = node;
    }

#ifdef lint
  result.i = 0;
  result.j = 0;
  result.count = 0;
#endif

  return result;
}

static bool
gl_linked_iterator_next (gl_list_iterator_t *iterator,
                         const void **eltp, gl_list_node_t *nodep)
{
  if (iterator->p != iterator->q)
    {
      gl_list_node_t node = (gl_list_node_t) iterator->p;
      *eltp = node->value;
      if (nodep != NULL)
        *nodep = node;
      iterator->p = node->next;
      return true;
    }
  else
    return false;
}

static void
gl_linked_iterator_free (gl_list_iterator_t *iterator)
{
}

/* ---------------------- Sorted gl_list_t Data Type ---------------------- */

static gl_list_node_t
gl_linked_sortedlist_search (gl_list_t list, gl_listelement_compar_fn compar,
                             const void *elt)
{
  gl_list_node_t node;

  for (node = list->root.next; node != &list->root; node = node->next)
    {
      int cmp = compar (node->value, elt);

      if (cmp > 0)
        break;
      if (cmp == 0)
        return node;
    }
  return NULL;
}

static gl_list_node_t
gl_linked_sortedlist_search_from_to (gl_list_t list,
                                     gl_listelement_compar_fn compar,
                                     size_t low, size_t high,
                                     const void *elt)
{
  size_t count = list->count;

  if (!(low <= high && high <= list->count))
    /* Invalid arguments.  */
    abort ();

  high -= low;
  if (high > 0)
    {
      /* Here we know low < count.  */
      size_t position = low;
      gl_list_node_t node;

      if (position <= ((count - 1) / 2))
        {
          node = list->root.next;
          for (; position > 0; position--)
            node = node->next;
        }
      else
        {
          position = count - 1 - position;
          node = list->root.prev;
          for (; position > 0; position--)
            node = node->prev;
        }

      do
        {
          int cmp = compar (node->value, elt);

          if (cmp > 0)
            break;
          if (cmp == 0)
            return node;
          node = node->next;
        }
      while (--high > 0);
    }
  return NULL;
}

static size_t
gl_linked_sortedlist_indexof (gl_list_t list, gl_listelement_compar_fn compar,
                              const void *elt)
{
  gl_list_node_t node;
  size_t index;

  for (node = list->root.next, index = 0;
       node != &list->root;
       node = node->next, index++)
    {
      int cmp = compar (node->value, elt);

      if (cmp > 0)
        break;
      if (cmp == 0)
        return index;
    }
  return (size_t)(-1);
}

static size_t
gl_linked_sortedlist_indexof_from_to (gl_list_t list,
                                      gl_listelement_compar_fn compar,
                                      size_t low, size_t high,
                                      const void *elt)
{
  size_t count = list->count;

  if (!(low <= high && high <= list->count))
    /* Invalid arguments.  */
    abort ();

  high -= low;
  if (high > 0)
    {
      /* Here we know low < count.  */
      size_t index = low;
      size_t position = low;
      gl_list_node_t node;

      if (position <= ((count - 1) / 2))
        {
          node = list->root.next;
          for (; position > 0; position--)
            node = node->next;
        }
      else
        {
          position = count - 1 - position;
          node = list->root.prev;
          for (; position > 0; position--)
            node = node->prev;
        }

      do
        {
          int cmp = compar (node->value, elt);

          if (cmp > 0)
            break;
          if (cmp == 0)
            return index;
          node = node->next;
          index++;
        }
      while (--high > 0);
    }
  return (size_t)(-1);
}

static gl_list_node_t
gl_linked_sortedlist_nx_add (gl_list_t list, gl_listelement_compar_fn compar,
                             const void *elt)
{
  gl_list_node_t node;

  for (node = list->root.next; node != &list->root; node = node->next)
    if (compar (node->value, elt) >= 0)
      return gl_linked_nx_add_before (list, node, elt);
  return gl_linked_nx_add_last (list, elt);
}

static bool
gl_linked_sortedlist_remove (gl_list_t list, gl_listelement_compar_fn compar,
                             const void *elt)
{
  gl_list_node_t node;

  for (node = list->root.next; node != &list->root; node = node->next)
    {
      int cmp = compar (node->value, elt);

      if (cmp > 0)
        break;
      if (cmp == 0)
        return gl_linked_remove_node (list, node);
    }
  return false;
}
