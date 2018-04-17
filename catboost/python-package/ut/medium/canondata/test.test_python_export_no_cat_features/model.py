### Model data
class CatboostModel(object):
        tree_count = 2
        float_feature_count = 8
        binary_feature_count = 8
        border_counts = [1, 1, 1, 1, 1, 1, 1, 1]
        borders = [0.5,0.5,0.5,0.0511068,0.5,0.5,0.5,0.259879]
        tree_depth  = [6, 4]
        tree_splits = [6, 4, 3, 5, 2, 7, 6, 0, 1, 4]

        # Aggregated array of leaf values for trees. Each tree is represented by a separate line:
        leaf_values = [
        0.0008211111193150276, 0, 0.0002249999959287899, 0, 0.001252702682193469, 0, 0.0003937499928753823, 0, 0.000989974279526389, 0.001679999969601631, 0.000647499988283962, 0, 0.0007328258582250274, 0, 0.001145454524728385, 0, 0.0008999999837151595, 0, 0, 0, 0, 0, 0, 0, 0.001374999961815775, 0, 0, 0, 0.001399999974668026, 0, 0.0005249999905005097, 0, 0.001039998001456305, 0, 0.002099999962002039, 0.001259999977201223, 0.001890090150245136, 0.002099999962002039, 0.002390476385184691, 0.00347205903071033, 0.001214583574654528, 0.0004199999924004077, 0.001259999977201223, 0.003790092530154747, 0.001781553489653492, 0.002363889468771702, 0.002231249959627166, 0.003923684268934943, 0, 0, 0, 0, 0.002604545391452584, 0, 0, 0, 0, 0, 0, 0, 0.00275191166696851, 0.002099999962002039, 0, 0,
        0.0007538577474999658, 0, 0.001673253488408079, 0.002738652127683553, 0.0002184838828884162, 0, 0.0008825505085869142, 0, 0.0003120346099853925, 0, 0.00158349975932882, 0.00373994872438916, 0, 0, 0, 0
]

### Model applicator
def apply_catboost_model(float_features):
    model = CatboostModel

    binary_feature_index = 0
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
