"""CoreML export for CatBoost-MLX models."""

from typing import Dict, Any, List
from ._tree_utils import unfold_oblivious_tree


def export_coreml(model_data: dict, path: str) -> None:
    """Export a CatBoost-MLX model to CoreML (.mlmodel / .mlpackage) format.

    Parameters
    ----------
    model_data : dict
        The model JSON data (as loaded from the model file).
    path : str
        Output path. Should end with .mlmodel or .mlpackage.
    """
    try:
        import coremltools as ct
        from coremltools.models import MLModel
        from coremltools.proto import Model_pb2
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install it with: pip install coremltools>=7.0"
        )

    info = model_data["model_info"]
    features = model_data["features"]
    trees = model_data["trees"]
    loss_type = info["loss_type"].split(":")[0]
    approx_dim = info.get("approx_dimension", 1)
    num_classes = info.get("num_classes", 0)
    num_features = len(features)

    is_classifier = loss_type in ("logloss", "multiclass")

    feature_spec = [("features", ct.models.datatypes.Array(num_features))]

    # Build coremltools model spec
    if is_classifier:
        if loss_type == "multiclass":
            class_labels = list(range(num_classes))
        else:
            class_labels = [0, 1]

        builder = ct.models.tree_ensemble.TreeEnsembleClassifier(
            features=feature_spec,
            class_labels=class_labels,
            output_features="predicted_class",
        )

        # Set post-evaluation transform for probability
        builder.spec.treeEnsembleClassifier.postEvaluationTransform = (
            Model_pb2.TreeEnsemblePostEvaluationTransform.Value(
                "Classification_SoftMax"
            )
        )
    else:
        builder = ct.models.tree_ensemble.TreeEnsembleRegressor(
            features=feature_spec,
            target="prediction",
        )

    # Add trees
    for tree_idx, tree in enumerate(trees):
        nodes = unfold_oblivious_tree(tree, features, approx_dim)

        for node in nodes:
            if node["type"] == "branch":
                feat_idx = node["feature_idx"]
                threshold = node["threshold"]

                if node["is_one_hot"]:
                    builder.add_branch_node(
                        tree_id=tree_idx,
                        node_id=node["node_id"],
                        feature_index=feat_idx,
                        feature_value=threshold,
                        branch_mode="BranchOnValueEqual",
                        true_child_id=node["true_child"],
                        false_child_id=node["false_child"],
                    )
                else:
                    # Numeric: oblivious tree goes right when binVal > threshold.
                    # CoreML BranchOnValueGreaterThan: true when value > threshold.
                    builder.add_branch_node(
                        tree_id=tree_idx,
                        node_id=node["node_id"],
                        feature_index=feat_idx,
                        feature_value=threshold,
                        branch_mode="BranchOnValueGreaterThan",
                        true_child_id=node["true_child"],
                        false_child_id=node["false_child"],
                    )
            else:
                # Leaf node
                values = {k: v for k, v in enumerate(node["values"])}
                builder.add_leaf_node(
                    tree_id=tree_idx,
                    node_id=node["node_id"],
                    values=values,
                )

    ct.models.utils.save_spec(builder.spec, path)
