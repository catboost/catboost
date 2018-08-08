### Model data
class CatboostModel(object):
    tree_count = 2
    float_feature_count = 9
    binary_feature_count = 11
    border_counts = [1, 1, 1, 2, 1, 1, 1, 1, 2]
    borders = [0.8374055, 0.60392153, 0.387779, 0.58333349, 1.5, 0.93881702, 0.061012201, 0.5, 0.97901797, 0.27336848, 0.66261351]
    tree_depth  = [6, 5]
    tree_splits = [10, 5, 0, 4, 1, 9, 3, 6, 8, 7, 2]

    # Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0005814876058138907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001133998390287161, 0, 0.0009975000284612179, 0, 0, 0, 0.0008555554668419063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001372431288473308, 0.002215142827481031, 0.001875349087640643, 0.002333333250135183, 0, 0, 0.001049999962560833, 0.006780804600566626, 0, 0, 0, 0, 0, 0, 0, 0, 0.002136547351256013, 0.002390978159382939, 0.002491590334102511, 0.003389361780136824, 0, 0, 0.002099999925121665, 0.003947368357330561, 0, 0, 0, 0.002099999925121665, 0, 0, 0, 0,
        0.001135568716563284, 0.001330881263129413, 0, 0, 0.0002743172226473689, 0, 0, 0, 0.001436591031961143, 0.001021189265884459, 0.003254958894103765, 0.003336918773129582, 0.000900790560990572, 0, 0.001032500062137842, 0, 0.002713314490392804, -2.869173840736039e-05, 0, 0, 0, 0, 0, 0, 0.00300825503654778, 0, 0.002350773196667433, 0.000385663821361959, 0, 0, 0, 0
]

### Model applicator
def apply_catboost_model(float_features):
    model = CatboostModel

    binary_feature_index = 0
    binary_features = [0] * model.binary_feature_count
    for i in range(model.float_feature_count):
        for j in range(model.border_counts[i]):
            binary_features[binary_feature_index] = 1 if (float_features[i] > model.borders[binary_feature_index]) else 0
            binary_feature_index += 1

    # Extract and sum values from trees
    result = 0.0
    tree_splits_index = 0
    current_tree_leaf_values_index = 0
    for tree_id in range(model.tree_count):
        current_tree_depth = model.tree_depth[tree_id]
        index = 0
        for depth in range(current_tree_depth):
            index |= (binary_features[model.tree_splits[tree_splits_index + depth]] << depth)
        result += model.leaf_values[current_tree_leaf_values_index + index]
        tree_splits_index += current_tree_depth
        current_tree_leaf_values_index += (1 << current_tree_depth)
    return result
