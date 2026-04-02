"""
export_onnx.py -- Exports trained models to the ONNX format for cross-platform inference.

What this file does:
    ONNX is like a universal file format for ML models -- think of it as PDF
    for documents. Any platform that reads ONNX (Windows, Linux, mobile, cloud)
    can run your model. This file translates our trained tree model into ONNX's
    TreeEnsembleRegressor or TreeEnsembleClassifier operators.

How it fits into the project:
    Called by core.py when users run model.export_onnx(). Imports _tree_utils.py
    to convert oblivious trees to standard binary trees. Requires the optional
    ``onnx`` pip package.

Key concepts:
    - ONNX: Open Neural Network Exchange -- an open standard for ML model files.
    - TreeEnsembleRegressor / TreeEnsembleClassifier: ONNX operators specifically
      designed to represent decision-tree-based models as flat node arrays.
    - Post-transform: ONNX's built-in final function (SOFTMAX for multiclass,
      LOGISTIC for binary). Poisson/Tweedie need a separate Exp node.
"""

from typing import List

from ._tree_utils import unfold_oblivious_tree


def export_onnx(model_data: dict, path: str) -> None:
    """Export a CatBoost-MLX model to ONNX format.

    Parameters
    ----------
    model_data : dict
        The model JSON data (as loaded from the model file).
    path : str
        Output path. Should end with .onnx.
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. "
            "Install it with: pip install onnx>=1.14"
        )

    info = model_data["model_info"]
    features = model_data["features"]
    trees = model_data["trees"]
    loss_type = info["loss_type"].split(":")[0]
    approx_dim = info.get("approx_dimension", 1)
    num_classes = info.get("num_classes", 0)
    num_features = len(features)

    is_classifier = loss_type in ("logloss", "multiclass")

    # ONNX TreeEnsemble operators require all tree data as flat parallel arrays
    # (one entry per node across ALL trees). We build these as we iterate.
    nodes_treeids: List[int] = []
    nodes_nodeids: List[int] = []
    nodes_featureids: List[int] = []
    nodes_values: List[float] = []
    nodes_modes: List[str] = []
    nodes_truenodeids: List[int] = []
    nodes_falsenodeids: List[int] = []
    nodes_hitrates: List[float] = []
    nodes_missing_value_tracks_true: List[int] = []

    # Target arrays (for leaf values)
    target_treeids: List[int] = []
    target_nodeids: List[int] = []
    target_ids: List[int] = []
    target_weights: List[float] = []

    # Class arrays (for classifier)
    class_treeids: List[int] = []
    class_nodeids: List[int] = []
    class_ids: List[int] = []
    class_weights: List[float] = []

    for tree_idx, tree in enumerate(trees):
        nodes = unfold_oblivious_tree(tree, features, approx_dim)

        for node in nodes:
            nid = node["node_id"]
            nodes_treeids.append(tree_idx)
            nodes_nodeids.append(nid)
            nodes_hitrates.append(1.0)
            nodes_missing_value_tracks_true.append(0)

            if node["type"] == "branch":
                if node["is_one_hot"]:
                    nodes_modes.append("BRANCH_EQ")
                else:
                    nodes_modes.append("BRANCH_GT")
                nodes_featureids.append(node["feature_idx"])
                nodes_values.append(node["threshold"])
                nodes_truenodeids.append(node["true_child"])
                nodes_falsenodeids.append(node["false_child"])
            else:
                nodes_modes.append("LEAF")
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)

                values = node["values"]
                if is_classifier:
                    for k, v in enumerate(values):
                        class_treeids.append(tree_idx)
                        class_nodeids.append(nid)
                        class_ids.append(k)
                        class_weights.append(v)
                else:
                    for k, v in enumerate(values):
                        target_treeids.append(tree_idx)
                        target_nodeids.append(nid)
                        target_ids.append(k)
                        target_weights.append(v)

    # Input
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, num_features])

    # ONNX uses different operators for classifiers vs regressors
    if is_classifier:
        n_targets = num_classes if loss_type == "multiclass" else 2
        class_labels = list(range(n_targets))

        if loss_type == "multiclass":
            post_transform = "SOFTMAX"
        else:
            post_transform = "LOGISTIC"

        # TreeEnsembleClassifier
        tree_node = helper.make_node(
            "TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["label", "probabilities"],
            domain="ai.onnx.ml",
            name="TreeEnsembleClassifier",
            nodes_treeids=nodes_treeids,
            nodes_nodeids=nodes_nodeids,
            nodes_featureids=nodes_featureids,
            nodes_values=nodes_values,
            nodes_modes=nodes_modes,
            nodes_truenodeids=nodes_truenodeids,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_hitrates=nodes_hitrates,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            class_treeids=class_treeids,
            class_nodeids=class_nodeids,
            class_ids=class_ids,
            class_weights=class_weights,
            classlabels_int64s=class_labels,
            post_transform=post_transform,
        )

        label_out = helper.make_tensor_value_info("label", TensorProto.INT64, [None])
        prob_out = helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, n_targets]
        )

        graph = helper.make_graph(
            [tree_node],
            "catboost_mlx_classifier",
            [X],
            [label_out, prob_out],
        )

    else:
        # Regressor: poisson/tweedie need Exp transform after tree evaluation,
        # but ONNX's TreeEnsembleRegressor has no built-in Exp. So we add a
        # separate Exp node in the ONNX graph.
        if loss_type in ("poisson", "tweedie"):
            # No built-in exp in TreeEnsembleRegressor, add Exp node
            post_transform = "NONE"
            needs_exp = True
        else:
            post_transform = "NONE"
            needs_exp = False

        tree_output_name = "raw_prediction" if needs_exp else "prediction"

        tree_node = helper.make_node(
            "TreeEnsembleRegressor",
            inputs=["X"],
            outputs=[tree_output_name],
            domain="ai.onnx.ml",
            name="TreeEnsembleRegressor",
            nodes_treeids=nodes_treeids,
            nodes_nodeids=nodes_nodeids,
            nodes_featureids=nodes_featureids,
            nodes_values=nodes_values,
            nodes_modes=nodes_modes,
            nodes_truenodeids=nodes_truenodeids,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_hitrates=nodes_hitrates,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            target_treeids=target_treeids,
            target_nodeids=target_nodeids,
            target_ids=target_ids,
            target_weights=target_weights,
            post_transform=post_transform,
            n_targets=approx_dim,
        )

        graph_nodes = [tree_node]

        if needs_exp:
            exp_node = helper.make_node("Exp", inputs=["raw_prediction"],
                                        outputs=["prediction"], name="Exp")
            graph_nodes.append(exp_node)

        pred_out = helper.make_tensor_value_info(
            "prediction", TensorProto.FLOAT, [None, approx_dim]
        )

        graph = helper.make_graph(
            graph_nodes,
            "catboost_mlx_regressor",
            [X],
            [pred_out],
        )

    # Create model with ai.onnx.ml opset
    opset_imports = [
        helper.make_opsetid("", 17),      # default ONNX opset
        helper.make_opsetid("ai.onnx.ml", 3),  # ML opset
    ]
    onnx_model = helper.make_model(graph, opset_imports=opset_imports)
    onnx_model.ir_version = 8

    # Validate
    onnx.checker.check_model(onnx_model)

    # Save
    onnx.save(onnx_model, path)
