# Choosing the tree structure

This is a greedy method. Features are selected in order along with their splits for substitution in each leaf. Candidates are selected based on data from the [preliminary calculation of splits](algorithm-main-stages_pre-count.md) and the [transformation of categorical features to numerical features](algorithm-main-stages_cat-to-numberic.md). The tree depth and other rules for choosing the structure are set in the starting parameters.

How a <q>feature-split</q> pair is chosen for a leaf:
1. A list is formed of possible candidates (<q>feature-split pairs</q>) to be assigned to a leaf as the split.

1. A number of penalty functions are calculated for each object (on the condition that all of the candidates obtained from step [1](#candidates) have been assigned to the leaf).

1. The split with the smallest penalty is selected.

The resulting value is assigned to the leaf.

This procedure is repeated for all following leaves (the number of leaves needs to match the depth of the tree).

Before building each new tree, a random permutation of classification objects is performed. A [metric](loss-functions.md), which determines the direction for further improving the function, is used to select the structure of the next tree. The value is calculated sequentially for each object. The permutation obtained before building the tree is used in the calculation – the data for the objects are used in the order in which they were placed before the procedure.
