/* Sequential list data type implemented by a binary tree.
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

/* Common code of gl_rbtree_list.c and gl_rbtreehash_list.c.  */

/* -------------------------- gl_list_t Data Type -------------------------- */

/* Creates a subtree for count >= 1 elements.
   Its black-height bh is passed as argument, with
   2^bh - 1 <= count <= 2^(bh+1) - 1.  bh == 0 implies count == 1.
   Its height is h where 2^(h-1) <= count <= 2^h - 1.
   Return NULL upon out-of-memory.  */
static gl_list_node_t
create_subtree_with_contents (unsigned int bh,
                              size_t count, const void **contents)
{
  size_t half1 = (count - 1) / 2;
  size_t half2 = count / 2;
  /* Note: half1 + half2 = count - 1.  */
  gl_list_node_t node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));
  if (node == NULL)
    return NULL;

  if (half1 > 0)
    {
      /* half1 > 0 implies count > 1, implies bh >= 1, implies
           2^(bh-1) - 1 <= half1 <= 2^bh - 1.  */
      node->left =
        create_subtree_with_contents (bh - 1, half1, contents);
      if (node->left == NULL)
        goto fail1;
      node->left->parent = node;
    }
  else
    node->left = NULL;

  node->value = contents[half1];

  if (half2 > 0)
    {
      /* half2 > 0 implies count > 1, implies bh >= 1, implies
           2^(bh-1) - 1 <= half2 <= 2^bh - 1.  */
      node->right =
       create_subtree_with_contents (bh - 1, half2, contents + half1 + 1);
      if (node->right == NULL)
        goto fail2;
      node->right->parent = node;
    }
  else
    node->right = NULL;

  node->color = (bh == 0 ? RED : BLACK);

  node->branch_size = count;

  return node;

 fail2:
  if (node->left != NULL)
    free_subtree (node->left);
 fail1:
  free (node);
  return NULL;
}

static gl_list_t
gl_tree_nx_create (gl_list_implementation_t implementation,
                   gl_listelement_equals_fn equals_fn,
                   gl_listelement_hashcode_fn hashcode_fn,
                   gl_listelement_dispose_fn dispose_fn,
                   bool allow_duplicates,
                   size_t count, const void **contents)
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
  if (count > 0)
    {
      /* Assuming 2^bh - 1 <= count <= 2^(bh+1) - 2, we create a tree whose
         upper bh levels are black, and only the partially present lowest
         level is red.  */
      unsigned int bh;
      {
        size_t n;
        for (n = count + 1, bh = 0; n > 1; n = n >> 1)
          bh++;
      }

      list->root = create_subtree_with_contents (bh, count, contents);
      if (list->root == NULL)
        goto fail2;
      list->root->parent = NULL;

#if WITH_HASHTABLE
      /* Now that the tree is built, node_position() works.  Now we can
         add the nodes to the hash table.  */
      if (add_nodes_to_buckets (list) < 0)
        goto fail3;
#endif
    }
  else
    list->root = NULL;

  return list;

#if WITH_HASHTABLE
 fail3:
  free_subtree (list->root);
#endif
 fail2:
#if WITH_HASHTABLE
  free (list->table);
 fail1:
#endif
  free (list);
  return NULL;
}

/* Rotates left a subtree.

                         B                         D
                       /   \                     /   \
                     A       D       -->       B       E
                            / \               / \
                           C   E             A   C

   Changes the tree structure, updates the branch sizes.
   The caller must update the colors and register D as child of its parent.  */
static gl_list_node_t
rotate_left (gl_list_node_t b_node, gl_list_node_t d_node)
{
  gl_list_node_t a_node = b_node->left;
  gl_list_node_t c_node = d_node->left;
  gl_list_node_t e_node = d_node->right;

  b_node->right = c_node;
  d_node->left = b_node;

  d_node->parent = b_node->parent;
  b_node->parent = d_node;
  if (c_node != NULL)
    c_node->parent = b_node;

  b_node->branch_size =
    (a_node != NULL ? a_node->branch_size : 0)
    + 1 + (c_node != NULL ? c_node->branch_size : 0);
  d_node->branch_size =
    b_node->branch_size + 1 + (e_node != NULL ? e_node->branch_size : 0);

  return d_node;
}

/* Rotates right a subtree.

                           D                     B
                         /   \                 /   \
                       B       E     -->     A       D
                      / \                           / \
                     A   C                         C   E

   Changes the tree structure, updates the branch sizes.
   The caller must update the colors and register B as child of its parent.  */
static gl_list_node_t
rotate_right (gl_list_node_t b_node, gl_list_node_t d_node)
{
  gl_list_node_t a_node = b_node->left;
  gl_list_node_t c_node = b_node->right;
  gl_list_node_t e_node = d_node->right;

  d_node->left = c_node;
  b_node->right = d_node;

  b_node->parent = d_node->parent;
  d_node->parent = b_node;
  if (c_node != NULL)
    c_node->parent = d_node;

  d_node->branch_size =
    (c_node != NULL ? c_node->branch_size : 0)
    + 1 + (e_node != NULL ? e_node->branch_size : 0);
  b_node->branch_size =
    (a_node != NULL ? a_node->branch_size : 0) + 1 + d_node->branch_size;

  return b_node;
}

/* Ensures the tree is balanced, after an insertion operation.
   Also assigns node->color.
   parent is the given node's parent, known to be non-NULL.  */
static void
rebalance_after_add (gl_list_t list, gl_list_node_t node, gl_list_node_t parent)
{
  for (;;)
    {
      /* At this point, parent = node->parent != NULL.
         Think of node->color being RED (although node->color is not yet
         assigned.)  */
      gl_list_node_t grandparent;
      gl_list_node_t uncle;

      if (parent->color == BLACK)
        {
          /* A RED color for node is acceptable.  */
          node->color = RED;
          return;
        }

      grandparent = parent->parent;
      /* Since parent is RED, we know that
         grandparent is != NULL and colored BLACK.  */

      if (grandparent->left == parent)
        uncle = grandparent->right;
      else if (grandparent->right == parent)
        uncle = grandparent->left;
      else
        abort ();

      if (uncle != NULL && uncle->color == RED)
        {
          /* Change grandparent from BLACK to RED, and
             change parent and uncle from RED to BLACK.
             This makes it acceptable for node to be RED.  */
          node->color = RED;
          parent->color = uncle->color = BLACK;
          node = grandparent;
        }
      else
        {
          /* grandparent and uncle are BLACK.  parent is RED.  node wants
             to be RED too.
             In this case, recoloring is not sufficient.  Need to perform
             one or two rotations.  */
          gl_list_node_t *grandparentp;

          if (grandparent->parent == NULL)
            grandparentp = &list->root;
          else if (grandparent->parent->left == grandparent)
            grandparentp = &grandparent->parent->left;
          else if (grandparent->parent->right == grandparent)
            grandparentp = &grandparent->parent->right;
          else
            abort ();

          if (grandparent->left == parent)
            {
              if (parent->right == node)
                {
                  /* Rotation between node and parent.  */
                  grandparent->left = rotate_left (parent, node);
                  node = parent;
                  parent = grandparent->left;
                }
              /* grandparent and uncle are BLACK.  parent and node want to be
                 RED.  parent = grandparent->left.  node = parent->left.

                      grandparent              parent
                         bh+1                   bh+1
                         /   \                 /   \
                     parent  uncle    -->   node  grandparent
                      bh      bh             bh      bh
                      / \                           / \
                   node  C                         C  uncle
                    bh   bh                       bh    bh
               */
              *grandparentp = rotate_right (parent, grandparent);
              parent->color = BLACK;
              node->color = grandparent->color = RED;
            }
          else /* grandparent->right == parent */
            {
              if (parent->left == node)
                {
                  /* Rotation between node and parent.  */
                  grandparent->right = rotate_right (node, parent);
                  node = parent;
                  parent = grandparent->right;
                }
              /* grandparent and uncle are BLACK.  parent and node want to be
                 RED.  parent = grandparent->right.  node = parent->right.

                    grandparent                    parent
                       bh+1                         bh+1
                       /   \                       /   \
                   uncle  parent     -->   grandparent  node
                     bh     bh                  bh       bh
                            / \                 / \
                           C  node          uncle  C
                          bh   bh            bh    bh
               */
              *grandparentp = rotate_left (grandparent, parent);
              parent->color = BLACK;
              node->color = grandparent->color = RED;
            }
          return;
        }

      /* Start again with a new (node, parent) pair.  */
      parent = node->parent;

      if (parent == NULL)
        {
          /* Change node's color from RED to BLACK.  This increases the
             tree's black-height.  */
          node->color = BLACK;
          return;
        }
    }
}

/* Ensures the tree is balanced, after a deletion operation.
   CHILD was a grandchild of PARENT and is now its child.  Between them,
   a black node was removed.  CHILD is also black, or NULL.
   (CHILD can also be NULL.  But PARENT is non-NULL.)  */
static void
rebalance_after_remove (gl_list_t list, gl_list_node_t child, gl_list_node_t parent)
{
  for (;;)
    {
      /* At this point, we reduced the black-height of the CHILD subtree by 1.
         To make up, either look for a possibility to turn a RED to a BLACK
         node, or try to reduce the black-height tree of CHILD's sibling
         subtree as well.  */
      gl_list_node_t *parentp;

      if (parent->parent == NULL)
        parentp = &list->root;
      else if (parent->parent->left == parent)
        parentp = &parent->parent->left;
      else if (parent->parent->right == parent)
        parentp = &parent->parent->right;
      else
        abort ();

      if (parent->left == child)
        {
          gl_list_node_t sibling = parent->right;
          /* sibling's black-height is >= 1.  In particular,
             sibling != NULL.

                      parent
                       /   \
                   child  sibling
                     bh    bh+1
           */

          if (sibling->color == RED)
            {
              /* sibling is RED, hence parent is BLACK and sibling's children
                 are non-NULL and BLACK.

                      parent                       sibling
                       bh+2                         bh+2
                       /   \                        /   \
                   child  sibling     -->       parent    SR
                     bh    bh+1                  bh+1    bh+1
                            / \                  / \
                          SL   SR            child  SL
                         bh+1 bh+1             bh  bh+1
               */
              *parentp = rotate_left (parent, sibling);
              parent->color = RED;
              sibling->color = BLACK;

              /* Concentrate on the subtree of parent.  The new sibling is
                 one of the old sibling's children, and known to be BLACK.  */
              parentp = &sibling->left;
              sibling = parent->right;
            }
          /* Now we know that sibling is BLACK.

                      parent
                       /   \
                   child  sibling
                     bh    bh+1
           */
          if (sibling->right != NULL && sibling->right->color == RED)
            {
              /*
                      parent                     sibling
                     bh+1|bh+2                  bh+1|bh+2
                       /   \                      /   \
                   child  sibling    -->      parent    SR
                     bh    bh+1                bh+1    bh+1
                            / \                / \
                          SL   SR           child  SL
                          bh   bh             bh   bh
               */
              *parentp = rotate_left (parent, sibling);
              sibling->color = parent->color;
              parent->color = BLACK;
              sibling->right->color = BLACK;
              return;
            }
          else if (sibling->left != NULL && sibling->left->color == RED)
            {
              /*
                      parent                   parent
                     bh+1|bh+2                bh+1|bh+2
                       /   \                    /   \
                   child  sibling    -->    child    SL
                     bh    bh+1               bh    bh+1
                            / \                     /  \
                          SL   SR                 SLL  sibling
                          bh   bh                 bh     bh
                         /  \                           /   \
                       SLL  SLR                       SLR    SR
                       bh    bh                       bh     bh

                 where SLL, SLR, SR are all black.
               */
              parent->right = rotate_right (sibling->left, sibling);
              /* Change sibling from BLACK to RED and SL from RED to BLACK.  */
              sibling->color = RED;
              sibling = parent->right;
              sibling->color = BLACK;

              /* Now do as in the previous case.  */
              *parentp = rotate_left (parent, sibling);
              sibling->color = parent->color;
              parent->color = BLACK;
              sibling->right->color = BLACK;
              return;
            }
          else
            {
              if (parent->color == BLACK)
                {
                  /* Change sibling from BLACK to RED.  Then the entire
                     subtree at parent has decreased its black-height.
                              parent                   parent
                               bh+2                     bh+1
                               /   \                    /   \
                           child  sibling    -->    child  sibling
                             bh    bh+1               bh     bh
                   */
                  sibling->color = RED;

                  child = parent;
                }
              else
                {
                  /* Change parent from RED to BLACK, but compensate by
                     changing sibling from BLACK to RED.
                              parent                   parent
                               bh+1                     bh+1
                               /   \                    /   \
                           child  sibling    -->    child  sibling
                             bh    bh+1               bh     bh
                   */
                  parent->color = BLACK;
                  sibling->color = RED;
                  return;
                }
            }
        }
      else if (parent->right == child)
        {
          gl_list_node_t sibling = parent->left;
          /* sibling's black-height is >= 1.  In particular,
             sibling != NULL.

                      parent
                       /   \
                  sibling  child
                    bh+1     bh
           */

          if (sibling->color == RED)
            {
              /* sibling is RED, hence parent is BLACK and sibling's children
                 are non-NULL and BLACK.

                      parent                 sibling
                       bh+2                    bh+2
                       /   \                  /   \
                  sibling  child    -->     SR    parent
                    bh+1     ch            bh+1    bh+1
                    / \                            / \
                  SL   SR                        SL  child
                 bh+1 bh+1                      bh+1   bh
               */
              *parentp = rotate_right (sibling, parent);
              parent->color = RED;
              sibling->color = BLACK;

              /* Concentrate on the subtree of parent.  The new sibling is
                 one of the old sibling's children, and known to be BLACK.  */
              parentp = &sibling->right;
              sibling = parent->left;
            }
          /* Now we know that sibling is BLACK.

                      parent
                       /   \
                  sibling  child
                    bh+1     bh
           */
          if (sibling->left != NULL && sibling->left->color == RED)
            {
              /*
                       parent                 sibling
                      bh+1|bh+2              bh+1|bh+2
                        /   \                  /   \
                   sibling  child    -->     SL   parent
                     bh+1     bh            bh+1   bh+1
                     / \                           / \
                   SL   SR                       SR  child
                   bh   bh                       bh    bh
               */
              *parentp = rotate_right (sibling, parent);
              sibling->color = parent->color;
              parent->color = BLACK;
              sibling->left->color = BLACK;
              return;
            }
          else if (sibling->right != NULL && sibling->right->color == RED)
            {
              /*
                      parent                       parent
                     bh+1|bh+2                    bh+1|bh+2
                       /   \                        /   \
                   sibling  child    -->          SR    child
                    bh+1      bh                 bh+1     bh
                     / \                         /  \
                   SL   SR                  sibling  SRR
                   bh   bh                    bh      bh
                       /  \                  /   \
                     SRL  SRR               SL   SRL
                     bh    bh               bh    bh

                 where SL, SRL, SRR are all black.
               */
              parent->left = rotate_left (sibling, sibling->right);
              /* Change sibling from BLACK to RED and SL from RED to BLACK.  */
              sibling->color = RED;
              sibling = parent->left;
              sibling->color = BLACK;

              /* Now do as in the previous case.  */
              *parentp = rotate_right (sibling, parent);
              sibling->color = parent->color;
              parent->color = BLACK;
              sibling->left->color = BLACK;
              return;
            }
          else
            {
              if (parent->color == BLACK)
                {
                  /* Change sibling from BLACK to RED.  Then the entire
                     subtree at parent has decreased its black-height.
                              parent                   parent
                               bh+2                     bh+1
                               /   \                    /   \
                           sibling  child    -->    sibling  child
                            bh+1      bh              bh       bh
                   */
                  sibling->color = RED;

                  child = parent;
                }
              else
                {
                  /* Change parent from RED to BLACK, but compensate by
                     changing sibling from BLACK to RED.
                              parent                   parent
                               bh+1                     bh+1
                               /   \                    /   \
                           sibling  child    -->    sibling  child
                            bh+1      bh              bh       bh
                   */
                  parent->color = BLACK;
                  sibling->color = RED;
                  return;
                }
            }
        }
      else
        abort ();

      /* Start again with a new (child, parent) pair.  */
      parent = child->parent;

#if 0 /* Already handled.  */
      if (child != NULL && child->color == RED)
        {
          child->color = BLACK;
          return;
        }
#endif

      if (parent == NULL)
        return;
    }
}

static void
gl_tree_remove_node_from_tree (gl_list_t list, gl_list_node_t node)
{
  gl_list_node_t parent = node->parent;

  if (node->left == NULL)
    {
      /* Replace node with node->right.  */
      gl_list_node_t child = node->right;

      if (child != NULL)
        {
          child->parent = parent;
          /* Since node->left == NULL, child must be RED and of height 1,
             hence node must have been BLACK.  Recolor the child.  */
          child->color = BLACK;
        }
      if (parent == NULL)
        list->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;

          /* Update branch_size fields of the parent nodes.  */
          {
            gl_list_node_t p;

            for (p = parent; p != NULL; p = p->parent)
              p->branch_size--;
          }

          if (child == NULL && node->color == BLACK)
            rebalance_after_remove (list, child, parent);
        }
    }
  else if (node->right == NULL)
    {
      /* It is not absolutely necessary to treat this case.  But the more
         general case below is more complicated, hence slower.  */
      /* Replace node with node->left.  */
      gl_list_node_t child = node->left;

      child->parent = parent;
      /* Since node->right == NULL, child must be RED and of height 1,
         hence node must have been BLACK.  Recolor the child.  */
      child->color = BLACK;
      if (parent == NULL)
        list->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;

          /* Update branch_size fields of the parent nodes.  */
          {
            gl_list_node_t p;

            for (p = parent; p != NULL; p = p->parent)
              p->branch_size--;
          }
        }
    }
  else
    {
      /* Replace node with the rightmost element of the node->left subtree.  */
      gl_list_node_t subst;
      gl_list_node_t subst_parent;
      gl_list_node_t child;
      color_t removed_color;

      for (subst = node->left; subst->right != NULL; )
        subst = subst->right;

      subst_parent = subst->parent;

      child = subst->left;

      removed_color = subst->color;

      /* The case subst_parent == node is special:  If we do nothing special,
         we get confusion about node->left, subst->left and child->parent.
           subst_parent == node
           <==> The 'for' loop above terminated immediately.
           <==> subst == subst_parent->left
                [otherwise subst == subst_parent->right]
         In this case, we would need to first set
           child->parent = node; node->left = child;
         and later - when we copy subst into node's position - again
           child->parent = subst; subst->left = child;
         Altogether a no-op.  */
      if (subst_parent != node)
        {
          if (child != NULL)
            child->parent = subst_parent;
          subst_parent->right = child;
        }

      /* Update branch_size fields of the parent nodes.  */
      {
        gl_list_node_t p;

        for (p = subst_parent; p != NULL; p = p->parent)
          p->branch_size--;
      }

      /* Copy subst into node's position.
         (This is safer than to copy subst's value into node, keep node in
         place, and free subst.)  */
      if (subst_parent != node)
        {
          subst->left = node->left;
          subst->left->parent = subst;
        }
      subst->right = node->right;
      subst->right->parent = subst;
      subst->color = node->color;
      subst->branch_size = node->branch_size;
      subst->parent = parent;
      if (parent == NULL)
        list->root = subst;
      else if (parent->left == node)
        parent->left = subst;
      else /* parent->right == node */
        parent->right = subst;

      if (removed_color == BLACK)
        {
          if (child != NULL && child->color == RED)
            /* Recolor the child.  */
            child->color = BLACK;
          else
            /* Rebalancing starts at child's parent, that is subst_parent -
               except when subst_parent == node.  In this case, we need to use
               its replacement, subst.  */
            rebalance_after_remove (list, child,
                                    subst_parent != node ? subst_parent : subst);
        }
    }
}

static gl_list_node_t
gl_tree_nx_add_first (gl_list_t list, const void *elt)
{
  /* Create new node.  */
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->branch_size = 1;
  new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);
#endif

  /* Add it to the tree.  */
  if (list->root == NULL)
    {
      new_node->color = BLACK;
      list->root = new_node;
      new_node->parent = NULL;
    }
  else
    {
      gl_list_node_t node;

      for (node = list->root; node->left != NULL; )
        node = node->left;

      node->left = new_node;
      new_node->parent = node;

      /* Update branch_size fields of the parent nodes.  */
      {
        gl_list_node_t p;

        for (p = node; p != NULL; p = p->parent)
          p->branch_size++;
      }

      /* Color and rebalance.  */
      rebalance_after_add (list, new_node, node);
    }

#if WITH_HASHTABLE
  /* Add node to the hash table.
     Note that this is only possible _after_ the node has been added to the
     tree structure, because add_to_bucket() uses node_position().  */
  if (add_to_bucket (list, new_node) < 0)
    {
      gl_tree_remove_node_from_tree (list, new_node);
      free (new_node);
      return NULL;
    }
  hash_resize_after_add (list);
#endif

  return new_node;
}

static gl_list_node_t
gl_tree_nx_add_last (gl_list_t list, const void *elt)
{
  /* Create new node.  */
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->branch_size = 1;
  new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);
#endif

  /* Add it to the tree.  */
  if (list->root == NULL)
    {
      new_node->color = BLACK;
      list->root = new_node;
      new_node->parent = NULL;
    }
  else
    {
      gl_list_node_t node;

      for (node = list->root; node->right != NULL; )
        node = node->right;

      node->right = new_node;
      new_node->parent = node;

      /* Update branch_size fields of the parent nodes.  */
      {
        gl_list_node_t p;

        for (p = node; p != NULL; p = p->parent)
          p->branch_size++;
      }

      /* Color and rebalance.  */
      rebalance_after_add (list, new_node, node);
    }

#if WITH_HASHTABLE
  /* Add node to the hash table.
     Note that this is only possible _after_ the node has been added to the
     tree structure, because add_to_bucket() uses node_position().  */
  if (add_to_bucket (list, new_node) < 0)
    {
      gl_tree_remove_node_from_tree (list, new_node);
      free (new_node);
      return NULL;
    }
  hash_resize_after_add (list);
#endif

  return new_node;
}

static gl_list_node_t
gl_tree_nx_add_before (gl_list_t list, gl_list_node_t node, const void *elt)
{
  /* Create new node.  */
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->branch_size = 1;
  new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);
#endif

  /* Add it to the tree.  */
  if (node->left == NULL)
    node->left = new_node;
  else
    {
      for (node = node->left; node->right != NULL; )
        node = node->right;
      node->right = new_node;
    }
  new_node->parent = node;

  /* Update branch_size fields of the parent nodes.  */
  {
    gl_list_node_t p;

    for (p = node; p != NULL; p = p->parent)
      p->branch_size++;
  }

  /* Color and rebalance.  */
  rebalance_after_add (list, new_node, node);

#if WITH_HASHTABLE
  /* Add node to the hash table.
     Note that this is only possible _after_ the node has been added to the
     tree structure, because add_to_bucket() uses node_position().  */
  if (add_to_bucket (list, new_node) < 0)
    {
      gl_tree_remove_node_from_tree (list, new_node);
      free (new_node);
      return NULL;
    }
  hash_resize_after_add (list);
#endif

  return new_node;
}

static gl_list_node_t
gl_tree_nx_add_after (gl_list_t list, gl_list_node_t node, const void *elt)
{
  /* Create new node.  */
  gl_list_node_t new_node =
    (struct gl_list_node_impl *) malloc (sizeof (struct gl_list_node_impl));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->branch_size = 1;
  new_node->value = elt;
#if WITH_HASHTABLE
  new_node->h.hashcode =
    (list->base.hashcode_fn != NULL
     ? list->base.hashcode_fn (new_node->value)
     : (size_t)(uintptr_t) new_node->value);
#endif

  /* Add it to the tree.  */
  if (node->right == NULL)
    node->right = new_node;
  else
    {
      for (node = node->right; node->left != NULL; )
        node = node->left;
      node->left = new_node;
    }
  new_node->parent = node;

  /* Update branch_size fields of the parent nodes.  */
  {
    gl_list_node_t p;

    for (p = node; p != NULL; p = p->parent)
      p->branch_size++;
  }

  /* Color and rebalance.  */
  rebalance_after_add (list, new_node, node);

#if WITH_HASHTABLE
  /* Add node to the hash table.
     Note that this is only possible _after_ the node has been added to the
     tree structure, because add_to_bucket() uses node_position().  */
  if (add_to_bucket (list, new_node) < 0)
    {
      gl_tree_remove_node_from_tree (list, new_node);
      free (new_node);
      return NULL;
    }
  hash_resize_after_add (list);
#endif

  return new_node;
}
