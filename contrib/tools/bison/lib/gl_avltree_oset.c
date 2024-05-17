/* Ordered set data type implemented by a binary tree.
   Copyright (C) 2006-2007, 2009-2013 Free Software Foundation, Inc.
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

#include <config.h>

/* Specification.  */
#include "gl_avltree_oset.h"

#include <stdlib.h>

/* An AVL tree is a binary tree where
   1. The height of each node is calculated as
        heightof(node) = 1 + max (heightof(node.left), heightof(node.right)).
   2. The heights of the subtrees of each node differ by at most 1:
        | heightof(right) - heightof(left) | <= 1.
   3. The index of the elements in the node.left subtree are smaller than
      the index of node.
      The index of the elements in the node.right subtree are larger than
      the index of node.
 */

/* -------------------------- gl_oset_t Data Type -------------------------- */

/* Tree node implementation, valid for this file only.  */
struct gl_oset_node_impl
{
  struct gl_oset_node_impl *left;   /* left branch, or NULL */
  struct gl_oset_node_impl *right;  /* right branch, or NULL */
  /* Parent pointer, or NULL. The parent pointer is not needed for most
     operations.  It is needed so that a gl_oset_node_t can be returned
     without memory allocation, on which the functions gl_oset_remove_node,
     gl_oset_add_before, gl_oset_add_after can be implemented.  */
  struct gl_oset_node_impl *parent;
  int balance;                      /* heightof(right) - heightof(left),
                                       always = -1 or 0 or 1 */
  const void *value;
};
typedef struct gl_oset_node_impl * gl_oset_node_t;

/* Concrete gl_oset_impl type, valid for this file only.  */
struct gl_oset_impl
{
  struct gl_oset_impl_base base;
  struct gl_oset_node_impl *root;   /* root node or NULL */
  size_t count;                     /* number of nodes */
};

/* An AVL tree of height h has at least F_(h+2) [Fibonacci number] and at most
   2^h - 1 elements.  So, h <= 84 (because a tree of height h >= 85 would have
   at least F_87 elements, and because even on 64-bit machines,
     sizeof (gl_oset_node_impl) * F_87 > 2^64
   this would exceed the address space of the machine.  */
#define MAXHEIGHT 83

/* Ensure the tree is balanced, after an insertion or deletion operation.
   The height of NODE is incremented by HEIGHT_DIFF (1 or -1).
   PARENT = NODE->parent.  (NODE can also be NULL.  But PARENT is non-NULL.)
   Rotation operations are performed starting at PARENT (not NODE itself!).  */
static void
rebalance (gl_oset_t set,
           gl_oset_node_t node, int height_diff, gl_oset_node_t parent)
{
  for (;;)
    {
      gl_oset_node_t child;
      int previous_balance;
      int balance_diff;
      gl_oset_node_t nodeleft;
      gl_oset_node_t noderight;

      child = node;
      node = parent;

      previous_balance = node->balance;

      /* The balance of NODE is incremented by BALANCE_DIFF: +1 if the right
         branch's height has increased by 1 or the left branch's height has
         decreased by 1, -1 if the right branch's height has decreased by 1 or
         the left branch's height has increased by 1, 0 if no height change.  */
      if (node->left != NULL || node->right != NULL)
        balance_diff = (child == node->right ? height_diff : -height_diff);
      else
        /* Special case where above formula doesn't work, because the caller
           didn't tell whether node's left or right branch shrunk from height 1
           to NULL.  */
        balance_diff = - previous_balance;

      node->balance += balance_diff;
      if (balance_diff == previous_balance)
        {
          /* node->balance is outside the range [-1,1].  Must rotate.  */
          gl_oset_node_t *nodep;

          if (node->parent == NULL)
            /* node == set->root */
            nodep = &set->root;
          else if (node->parent->left == node)
            nodep = &node->parent->left;
          else if (node->parent->right == node)
            nodep = &node->parent->right;
          else
            abort ();

          nodeleft = node->left;
          noderight = node->right;

          if (balance_diff < 0)
            {
              /* node->balance = -2.  The subtree is heavier on the left side.
                 Rotate from left to right:

                                  *
                                /   \
                             h+2      h
               */
              gl_oset_node_t nodeleftright = nodeleft->right;
              if (nodeleft->balance <= 0)
                {
                  /*
                              *                    h+2|h+3
                            /   \                  /    \
                         h+2      h      -->      /   h+1|h+2
                         / \                      |    /    \
                       h+1 h|h+1                 h+1  h|h+1  h
                   */
                  node->left = nodeleftright;
                  nodeleft->right = node;

                  nodeleft->parent = node->parent;
                  node->parent = nodeleft;
                  if (nodeleftright != NULL)
                    nodeleftright->parent = node;

                  nodeleft->balance += 1;
                  node->balance = - nodeleft->balance;

                  *nodep = nodeleft;
                  height_diff = (height_diff < 0
                                 ? /* noderight's height had been decremented from
                                      h+1 to h.  The subtree's height changes from
                                      h+3 to h+2|h+3.  */
                                   nodeleft->balance - 1
                                 : /* nodeleft's height had been incremented from
                                      h+1 to h+2.  The subtree's height changes from
                                      h+2 to h+2|h+3.  */
                                   nodeleft->balance);
                }
              else
                {
                  /*
                            *                     h+2
                          /   \                 /     \
                       h+2      h      -->    h+1     h+1
                       / \                    / \     / \
                      h  h+1                 h   L   R   h
                         / \
                        L   R

                   */
                  gl_oset_node_t L = nodeleft->right = nodeleftright->left;
                  gl_oset_node_t R = node->left = nodeleftright->right;
                  nodeleftright->left = nodeleft;
                  nodeleftright->right = node;

                  nodeleftright->parent = node->parent;
                  if (L != NULL)
                    L->parent = nodeleft;
                  if (R != NULL)
                    R->parent = node;
                  nodeleft->parent = nodeleftright;
                  node->parent = nodeleftright;

                  nodeleft->balance = (nodeleftright->balance > 0 ? -1 : 0);
                  node->balance = (nodeleftright->balance < 0 ? 1 : 0);
                  nodeleftright->balance = 0;

                  *nodep = nodeleftright;
                  height_diff = (height_diff < 0
                                 ? /* noderight's height had been decremented from
                                      h+1 to h.  The subtree's height changes from
                                      h+3 to h+2.  */
                                   -1
                                 : /* nodeleft's height had been incremented from
                                      h+1 to h+2.  The subtree's height changes from
                                      h+2 to h+2.  */
                                   0);
                }
            }
          else
            {
              /* node->balance = 2.  The subtree is heavier on the right side.
                 Rotate from right to left:

                                  *
                                /   \
                              h      h+2
               */
              gl_oset_node_t noderightleft = noderight->left;
              if (noderight->balance >= 0)
                {
                  /*
                              *                    h+2|h+3
                            /   \                   /    \
                          h      h+2     -->    h+1|h+2   \
                                 / \            /    \    |
                             h|h+1 h+1         h   h|h+1 h+1
                   */
                  node->right = noderightleft;
                  noderight->left = node;

                  noderight->parent = node->parent;
                  node->parent = noderight;
                  if (noderightleft != NULL)
                    noderightleft->parent = node;

                  noderight->balance -= 1;
                  node->balance = - noderight->balance;

                  *nodep = noderight;
                  height_diff = (height_diff < 0
                                 ? /* nodeleft's height had been decremented from
                                      h+1 to h.  The subtree's height changes from
                                      h+3 to h+2|h+3.  */
                                   - noderight->balance - 1
                                 : /* noderight's height had been incremented from
                                      h+1 to h+2.  The subtree's height changes from
                                      h+2 to h+2|h+3.  */
                                   - noderight->balance);
                }
              else
                {
                  /*
                            *                    h+2
                          /   \                /     \
                        h      h+2    -->    h+1     h+1
                               / \           / \     / \
                             h+1  h         h   L   R   h
                             / \
                            L   R

                   */
                  gl_oset_node_t L = node->right = noderightleft->left;
                  gl_oset_node_t R = noderight->left = noderightleft->right;
                  noderightleft->left = node;
                  noderightleft->right = noderight;

                  noderightleft->parent = node->parent;
                  if (L != NULL)
                    L->parent = node;
                  if (R != NULL)
                    R->parent = noderight;
                  node->parent = noderightleft;
                  noderight->parent = noderightleft;

                  node->balance = (noderightleft->balance > 0 ? -1 : 0);
                  noderight->balance = (noderightleft->balance < 0 ? 1 : 0);
                  noderightleft->balance = 0;

                  *nodep = noderightleft;
                  height_diff = (height_diff < 0
                                 ? /* nodeleft's height had been decremented from
                                      h+1 to h.  The subtree's height changes from
                                      h+3 to h+2.  */
                                   -1
                                 : /* noderight's height had been incremented from
                                      h+1 to h+2.  The subtree's height changes from
                                      h+2 to h+2.  */
                                   0);
                }
            }
          node = *nodep;
        }
      else
        {
          /* No rotation needed.  Only propagation of the height change to the
             next higher level.  */
          if (height_diff < 0)
            height_diff = (previous_balance == 0 ? 0 : -1);
          else
            height_diff = (node->balance == 0 ? 0 : 1);
        }

      if (height_diff == 0)
        break;

      parent = node->parent;
      if (parent == NULL)
        break;
    }
}

static gl_oset_node_t
gl_tree_nx_add_first (gl_oset_t set, const void *elt)
{
  /* Create new node.  */
  gl_oset_node_t new_node =
    (struct gl_oset_node_impl *) malloc (sizeof (struct gl_oset_node_impl));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->balance = 0;
  new_node->value = elt;

  /* Add it to the tree.  */
  if (set->root == NULL)
    {
      set->root = new_node;
      new_node->parent = NULL;
    }
  else
    {
      gl_oset_node_t node;

      for (node = set->root; node->left != NULL; )
        node = node->left;

      node->left = new_node;
      new_node->parent = node;
      node->balance--;

      /* Rebalance.  */
      if (node->right == NULL && node->parent != NULL)
        rebalance (set, node, 1, node->parent);
    }

  set->count++;
  return new_node;
}

static gl_oset_node_t
gl_tree_nx_add_before (gl_oset_t set, gl_oset_node_t node, const void *elt)
{
  /* Create new node.  */
  gl_oset_node_t new_node =
    (struct gl_oset_node_impl *) malloc (sizeof (struct gl_oset_node_impl));
  bool height_inc;

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->balance = 0;
  new_node->value = elt;

  /* Add it to the tree.  */
  if (node->left == NULL)
    {
      node->left = new_node;
      node->balance--;
      height_inc = (node->right == NULL);
    }
  else
    {
      for (node = node->left; node->right != NULL; )
        node = node->right;
      node->right = new_node;
      node->balance++;
      height_inc = (node->left == NULL);
    }
  new_node->parent = node;

  /* Rebalance.  */
  if (height_inc && node->parent != NULL)
    rebalance (set, node, 1, node->parent);

  set->count++;
  return new_node;
}

static gl_oset_node_t
gl_tree_nx_add_after (gl_oset_t set, gl_oset_node_t node, const void *elt)
{
  /* Create new node.  */
  gl_oset_node_t new_node =
    (struct gl_oset_node_impl *) malloc (sizeof (struct gl_oset_node_impl));
  bool height_inc;

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  new_node->balance = 0;
  new_node->value = elt;

  /* Add it to the tree.  */
  if (node->right == NULL)
    {
      node->right = new_node;
      node->balance++;
      height_inc = (node->left == NULL);
    }
  else
    {
      for (node = node->right; node->left != NULL; )
        node = node->left;
      node->left = new_node;
      node->balance--;
      height_inc = (node->right == NULL);
    }
  new_node->parent = node;

  /* Rebalance.  */
  if (height_inc && node->parent != NULL)
    rebalance (set, node, 1, node->parent);

  set->count++;
  return new_node;
}

static bool
gl_tree_remove_node (gl_oset_t set, gl_oset_node_t node)
{
  gl_oset_node_t parent = node->parent;

  if (node->left == NULL)
    {
      /* Replace node with node->right.  */
      gl_oset_node_t child = node->right;

      if (child != NULL)
        child->parent = parent;
      if (parent == NULL)
        set->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;

          rebalance (set, child, -1, parent);
        }
    }
  else if (node->right == NULL)
    {
      /* It is not absolutely necessary to treat this case.  But the more
         general case below is more complicated, hence slower.  */
      /* Replace node with node->left.  */
      gl_oset_node_t child = node->left;

      child->parent = parent;
      if (parent == NULL)
        set->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;

          rebalance (set, child, -1, parent);
        }
    }
  else
    {
      /* Replace node with the rightmost element of the node->left subtree.  */
      gl_oset_node_t subst;
      gl_oset_node_t subst_parent;
      gl_oset_node_t child;

      for (subst = node->left; subst->right != NULL; )
        subst = subst->right;

      subst_parent = subst->parent;

      child = subst->left;

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
      subst->balance = node->balance;
      subst->parent = parent;
      if (parent == NULL)
        set->root = subst;
      else if (parent->left == node)
        parent->left = subst;
      else /* parent->right == node */
        parent->right = subst;

      /* Rebalancing starts at child's parent, that is subst_parent -
         except when subst_parent == node.  In this case, we need to use
         its replacement, subst.  */
      rebalance (set, child, -1, subst_parent != node ? subst_parent : subst);
    }

  set->count--;
  if (set->base.dispose_fn != NULL)
    set->base.dispose_fn (node->value);
  free (node);
  return true;
}

/* Generic binary tree code.  */
#include "gl_anytree_oset.h"

/* For debugging.  */
static unsigned int
check_invariants (gl_oset_node_t node, gl_oset_node_t parent, size_t *counterp)
{
  unsigned int left_height =
    (node->left != NULL ? check_invariants (node->left, node, counterp) : 0);
  unsigned int right_height =
    (node->right != NULL ? check_invariants (node->right, node, counterp) : 0);
  int balance = (int)right_height - (int)left_height;

  if (!(node->parent == parent))
    abort ();
  if (!(balance >= -1 && balance <= 1))
    abort ();
  if (!(node->balance == balance))
    abort ();

  (*counterp)++;

  return 1 + (left_height > right_height ? left_height : right_height);
}
void
gl_avltree_oset_check_invariants (gl_oset_t set)
{
  size_t counter = 0;
  if (set->root != NULL)
    check_invariants (set->root, NULL, &counter);
  if (!(set->count == counter))
    abort ();
}

const struct gl_oset_implementation gl_avltree_oset_implementation =
  {
    gl_tree_nx_create_empty,
    gl_tree_size,
    gl_tree_search,
    gl_tree_search_atleast,
    gl_tree_nx_add,
    gl_tree_remove,
    gl_tree_oset_free,
    gl_tree_iterator,
    gl_tree_iterator_next,
    gl_tree_iterator_free
  };
