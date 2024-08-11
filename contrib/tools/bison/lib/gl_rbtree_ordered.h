/* Ordered {set,map} data type implemented by a binary tree.
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

/* Color of a node.  */
typedef enum color { BLACK, RED } color_t;

/* Tree node implementation, valid for this file only.  */
struct NODE_IMPL
{
  struct NODE_IMPL *left;   /* left branch, or NULL */
  struct NODE_IMPL *right;  /* right branch, or NULL */
  /* Parent pointer, or NULL. The parent pointer is not needed for most
     operations.  It is needed so that a NODE_T can be returned without
     memory allocation, on which the functions <container>_remove_node,
     <container>_add_before, <container>_add_after can be implemented.  */
  struct NODE_IMPL *parent;
  color_t color;                    /* node's color */
  NODE_PAYLOAD_FIELDS
};
typedef struct NODE_IMPL * NODE_T;

/* Concrete CONTAINER_IMPL type, valid for this file only.  */
struct CONTAINER_IMPL
{
  struct CONTAINER_IMPL_BASE base;
  struct NODE_IMPL *root;           /* root node or NULL */
  size_t count;                     /* number of nodes */
};

/* A red-black tree of height h has a black-height bh >= ceil(h/2) and
   therefore at least 2^ceil(h/2) - 1 elements.  So, h <= 116 (because a tree
   of height h >= 117 would have at least 2^59 - 1 elements, and because even
   on 64-bit machines,
     sizeof (NODE_IMPL) * (2^59 - 1) > 2^64
   this would exceed the address space of the machine.  */
#define MAXHEIGHT 116

/* Rotates left a subtree.

                         B                         D
                       /   \                     /   \
                     A       D       -->       B       E
                            / \               / \
                           C   E             A   C

   Changes the tree structure, updates the branch sizes.
   The caller must update the colors and register D as child of its parent.  */
static NODE_T
rotate_left (NODE_T b_node, NODE_T d_node)
{
  NODE_T c_node = d_node->left;

  b_node->right = c_node;
  d_node->left = b_node;

  d_node->parent = b_node->parent;
  b_node->parent = d_node;
  if (c_node != NULL)
    c_node->parent = b_node;

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
static NODE_T
rotate_right (NODE_T b_node, NODE_T d_node)
{
  NODE_T c_node = b_node->right;

  d_node->left = c_node;
  b_node->right = d_node;

  b_node->parent = d_node->parent;
  d_node->parent = b_node;
  if (c_node != NULL)
    c_node->parent = d_node;

  return b_node;
}

/* Ensures the tree is balanced, after an insertion operation.
   Also assigns node->color.
   parent is the given node's parent, known to be non-NULL.  */
static void
rebalance_after_add (CONTAINER_T container, NODE_T node, NODE_T parent)
{
  for (;;)
    {
      /* At this point, parent = node->parent != NULL.
         Think of node->color being RED (although node->color is not yet
         assigned.)  */
      NODE_T grandparent;
      NODE_T uncle;

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
          NODE_T *grandparentp;

          if (grandparent->parent == NULL)
            grandparentp = &container->root;
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
rebalance_after_remove (CONTAINER_T container, NODE_T child, NODE_T parent)
{
  for (;;)
    {
      /* At this point, we reduced the black-height of the CHILD subtree by 1.
         To make up, either look for a possibility to turn a RED to a BLACK
         node, or try to reduce the black-height tree of CHILD's sibling
         subtree as well.  */
      NODE_T *parentp;

      if (parent->parent == NULL)
        parentp = &container->root;
      else if (parent->parent->left == parent)
        parentp = &parent->parent->left;
      else if (parent->parent->right == parent)
        parentp = &parent->parent->right;
      else
        abort ();

      if (parent->left == child)
        {
          NODE_T sibling = parent->right;
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
          NODE_T sibling = parent->left;
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

static NODE_T
gl_tree_nx_add_first (CONTAINER_T container, NODE_PAYLOAD_PARAMS)
{
  /* Create new node.  */
  NODE_T new_node =
    (struct NODE_IMPL *) malloc (sizeof (struct NODE_IMPL));

  if (new_node == NULL)
    return NULL;

  new_node->left = NULL;
  new_node->right = NULL;
  NODE_PAYLOAD_ASSIGN(new_node)

  /* Add it to the tree.  */
  if (container->root == NULL)
    {
      new_node->color = BLACK;
      container->root = new_node;
      new_node->parent = NULL;
    }
  else
    {
      NODE_T node;

      for (node = container->root; node->left != NULL; )
        node = node->left;

      node->left = new_node;
      new_node->parent = node;

      /* Color and rebalance.  */
      rebalance_after_add (container, new_node, node);
    }

  container->count++;
  return new_node;
}

/* Adds the already allocated NEW_NODE to the tree, right before NODE.  */
static void
gl_tree_add_node_before (CONTAINER_T container, NODE_T node, NODE_T new_node)
{
  new_node->left = NULL;
  new_node->right = NULL;

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

  /* Color and rebalance.  */
  rebalance_after_add (container, new_node, node);

  container->count++;
}

static NODE_T
gl_tree_nx_add_before (CONTAINER_T container, NODE_T node, NODE_PAYLOAD_PARAMS)
{
  /* Create new node.  */
  NODE_T new_node =
    (struct NODE_IMPL *) malloc (sizeof (struct NODE_IMPL));

  if (new_node == NULL)
    return NULL;

  NODE_PAYLOAD_ASSIGN(new_node)

  gl_tree_add_node_before (container, node, new_node);
  return new_node;
}

/* Adds the already allocated NEW_NODE to the tree, right after NODE.  */
static void
gl_tree_add_node_after (CONTAINER_T container, NODE_T node, NODE_T new_node)
{
  new_node->left = NULL;
  new_node->right = NULL;

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

  /* Color and rebalance.  */
  rebalance_after_add (container, new_node, node);

  container->count++;
}

static NODE_T
gl_tree_nx_add_after (CONTAINER_T container, NODE_T node, NODE_PAYLOAD_PARAMS)
{
  /* Create new node.  */
  NODE_T new_node =
    (struct NODE_IMPL *) malloc (sizeof (struct NODE_IMPL));

  if (new_node == NULL)
    return NULL;

  NODE_PAYLOAD_ASSIGN(new_node)

  gl_tree_add_node_after (container, node, new_node);
  return new_node;
}

static void
gl_tree_remove_node_no_free (CONTAINER_T container, NODE_T node)
{
  NODE_T parent = node->parent;

  if (node->left == NULL)
    {
      /* Replace node with node->right.  */
      NODE_T child = node->right;

      if (child != NULL)
        {
          child->parent = parent;
          /* Since node->left == NULL, child must be RED and of height 1,
             hence node must have been BLACK.  Recolor the child.  */
          child->color = BLACK;
        }
      if (parent == NULL)
        container->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;

          if (child == NULL && node->color == BLACK)
            rebalance_after_remove (container, child, parent);
        }
    }
  else if (node->right == NULL)
    {
      /* It is not absolutely necessary to treat this case.  But the more
         general case below is more complicated, hence slower.  */
      /* Replace node with node->left.  */
      NODE_T child = node->left;

      child->parent = parent;
      /* Since node->right == NULL, child must be RED and of height 1,
         hence node must have been BLACK.  Recolor the child.  */
      child->color = BLACK;
      if (parent == NULL)
        container->root = child;
      else
        {
          if (parent->left == node)
            parent->left = child;
          else /* parent->right == node */
            parent->right = child;
        }
    }
  else
    {
      /* Replace node with the rightmost element of the node->left subtree.  */
      NODE_T subst;
      NODE_T subst_parent;
      NODE_T child;
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
      subst->parent = parent;
      if (parent == NULL)
        container->root = subst;
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
            rebalance_after_remove (container, child,
                                    subst_parent != node ? subst_parent : subst);
        }
    }

  container->count--;
}

static bool
gl_tree_remove_node (CONTAINER_T container, NODE_T node)
{
  gl_tree_remove_node_no_free (container, node);
  NODE_PAYLOAD_DISPOSE (container, node)
  free (node);
  return true;
}

/* For debugging.  */
static unsigned int
check_invariants (NODE_T node, NODE_T parent, size_t *counterp)
{
  unsigned int left_blackheight =
    (node->left != NULL ? check_invariants (node->left, node, counterp) : 0);
  unsigned int right_blackheight =
    (node->right != NULL ? check_invariants (node->right, node, counterp) : 0);

  if (!(node->parent == parent))
    abort ();
  if (!(node->color == BLACK || node->color == RED))
    abort ();
  if (parent == NULL && !(node->color == BLACK))
    abort ();
  if (!(left_blackheight == right_blackheight))
    abort ();

  (*counterp)++;

  return left_blackheight + (node->color == BLACK ? 1 : 0);
}
