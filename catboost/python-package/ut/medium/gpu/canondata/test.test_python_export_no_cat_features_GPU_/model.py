### Model data
class CatboostModel(object):
    tree_count = 2
    float_feature_count = 9
    binary_feature_count = 12
    border_counts = [1, 2, 1, 1, 3, 1, 1, 1, 1]
    borders = [0.011715351, 0.34408599, 0.76047051, 0.63000298, 0.43333352, 0.41666651, 0.58333349, 1.5, 0.94102502, 0.5, 0.50285947, 0.3318105]
    tree_depth  = [6, 6]
    tree_splits = [11, 2, 4, 7, 1, 8, 6, 3, 10, 9, 0, 5]

    # Aggregated array of leaf values for trees. Each tree is represented by a separate line:
    leaf_values = [
        0.0001500000071246177, 0.00201694923453033, 0, 0, 0.0005999999702908099, 0.002136326860636473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001308776205405593, 0.002161809476092458, 0.001329330960288644, 0.00378138036467135, 0.0004965625121258199, 0.00143311102874577, 0.0009444663301110268, 0.002024568850174546, 0, 0.00139999995008111, 0, 0.001049999962560833, 0, 0, 0, 0.001049999962560833, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.0004868714313488454, 0, 0, 0, 0, 0, 0, 0, 0.0007201300468295813, 0, 0, 0, 0.003045676508918405, 0, 0, 0, 0.001428714254871011, 0, 0.00415117060765624, 0, 0, 0, 0, 0, 0.001657033222727478, 0, 0.003496180754154921, 0, 0.003044427838176489, 0, -1.621357114345301e-05, 0, -7.083497166604502e-06, 0.0008241712930612266, 0, 0, 0, 0, 0, 0, -1.791286740626674e-05, 0, 0, 0, 0.0005087864119559526, 0.001163561129942536, 0, 0, 0.0006044265464879572, 0.001137889921665192, 0.0005150138749741018, -3.483569525997154e-05, 0, 0, 0, 0, 0.001425641938112676, 0.0004013623401988298, 0, 0, 0.00198999117128551, 0.003376733278855681, 0.0006148157408460975, 0
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
