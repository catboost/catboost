

__all__ = ['BallTree', 'BallTree64', 'BallTree32']

DOC_DICT64 = {
    'BinaryTree': 'BallTree64',
    'binary_tree': 'ball_tree64',
}

VALID_METRICS64 = [
    'BrayCurtisDistance64',
    'CanberraDistance64',
    'ChebyshevDistance64',
    'DiceDistance64',
    'EuclideanDistance64',
    'HammingDistance64',
    'HaversineDistance64',
    'JaccardDistance64',
    'MahalanobisDistance64',
    'ManhattanDistance64',
    'MinkowskiDistance64',
    'PyFuncDistance64',
    'RogersTanimotoDistance64',
    'RussellRaoDistance64',
    'SEuclideanDistance64',
    'SokalMichenerDistance64',
    'SokalSneathDistance64',
    'WMinkowskiDistance64',
]

DOC_DICT32 = {
    'BinaryTree': 'BallTree32',
    'binary_tree': 'ball_tree32',
}

VALID_METRICS32 = [
    'BrayCurtisDistance32',
    'CanberraDistance32',
    'ChebyshevDistance32',
    'DiceDistance32',
    'EuclideanDistance32',
    'HammingDistance32',
    'HaversineDistance32',
    'JaccardDistance32',
    'MahalanobisDistance32',
    'ManhattanDistance32',
    'MinkowskiDistance32',
    'PyFuncDistance32',
    'RogersTanimotoDistance32',
    'RussellRaoDistance32',
    'SEuclideanDistance32',
    'SokalMichenerDistance32',
    'SokalSneathDistance32',
    'WMinkowskiDistance32',
]

include "_binary_tree.pxi"

# Inherit BallTree64 from BinaryTree64
cdef class BallTree64(BinaryTree64):
    __doc__ = CLASS_DOC.format(**DOC_DICT64)
    pass

# Inherit BallTree32 from BinaryTree32
cdef class BallTree32(BinaryTree32):
    __doc__ = CLASS_DOC.format(**DOC_DICT32)
    pass


#----------------------------------------------------------------------
# The functions below specialized the Binary Tree as a Ball Tree
#
#   Note that these functions use the concept of "reduced distance".
#   The reduced distance, defined for some metrics, is a quantity which
#   is more efficient to compute than the distance, but preserves the
#   relative rankings of the true distance.  For example, the reduced
#   distance for the Euclidean metric is the squared-euclidean distance.
#   For some metrics, the reduced distance is simply the distance.

cdef int allocate_data64(
    BinaryTree64 tree,
    intp_t n_nodes,
    intp_t n_features,
) except -1:
    """Allocate arrays needed for the KD Tree"""
    tree.node_bounds = np.zeros((1, n_nodes, n_features), dtype=np.float64)
    return 0


cdef int init_node64(
    BinaryTree64 tree,
    NodeData_t[::1] node_data,
    intp_t i_node,
    intp_t idx_start,
    intp_t idx_end,
) except -1:
    """Initialize the node for the dataset stored in tree.data"""
    cdef intp_t n_features = tree.data.shape[1]
    cdef intp_t n_points = idx_end - idx_start

    cdef intp_t i, j
    cdef float64_t radius
    cdef const float64_t *this_pt

    cdef intp_t* idx_array = &tree.idx_array[0]
    cdef const float64_t* data = &tree.data[0, 0]
    cdef float64_t* centroid = &tree.node_bounds[0, i_node, 0]

    cdef bint with_sample_weight = tree.sample_weight is not None
    cdef const float64_t* sample_weight
    cdef float64_t sum_weight_node
    if with_sample_weight:
        sample_weight = &tree.sample_weight[0]

    # determine Node centroid
    for j in range(n_features):
        centroid[j] = 0

    if with_sample_weight:
        sum_weight_node = 0
        for i in range(idx_start, idx_end):
            sum_weight_node += sample_weight[idx_array[i]]
            this_pt = data + n_features * idx_array[i]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j] * sample_weight[idx_array[i]]

        for j in range(n_features):
            centroid[j] /= sum_weight_node
    else:
        for i in range(idx_start, idx_end):
            this_pt = data + n_features * idx_array[i]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j]

        for j in range(n_features):
            centroid[j] /= n_points

    # determine Node radius
    radius = 0
    for i in range(idx_start, idx_end):
        radius = fmax(radius,
                      tree.rdist(centroid,
                                 data + n_features * idx_array[i],
                                 n_features))

    node_data[i_node].radius = tree.dist_metric._rdist_to_dist(radius)
    node_data[i_node].idx_start = idx_start
    node_data[i_node].idx_end = idx_end
    return 0


cdef inline float64_t min_dist64(
    BinaryTree64 tree,
    intp_t i_node,
    const float64_t* pt,
) except -1 nogil:
    """Compute the minimum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    return fmax(0, dist_pt - tree.node_data[i_node].radius)


cdef inline float64_t max_dist64(
    BinaryTree64 tree,
    intp_t i_node,
    const float64_t* pt,
) except -1:
    """Compute the maximum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    return dist_pt + tree.node_data[i_node].radius


cdef inline int min_max_dist64(
    BinaryTree64 tree,
    intp_t i_node,
    const float64_t* pt,
    float64_t* min_dist,
    float64_t* max_dist,
) except -1 nogil:
    """Compute the minimum and maximum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    cdef float64_t rad = tree.node_data[i_node].radius
    min_dist[0] = fmax(0, dist_pt - rad)
    max_dist[0] = dist_pt + rad
    return 0


cdef inline float64_t min_rdist64(
    BinaryTree64 tree,
    intp_t i_node,
    const float64_t* pt,
) except -1 nogil:
    """Compute the minimum reduced-distance between a point and a node"""
    if tree.euclidean:
        return euclidean_dist_to_rdist64(
            min_dist64(tree, i_node, pt)
        )
    else:
        return tree.dist_metric._dist_to_rdist(
            min_dist64(tree, i_node, pt)
        )


cdef inline float64_t max_rdist64(
    BinaryTree64 tree,
    intp_t i_node,
    const float64_t* pt,
) except -1:
    """Compute the maximum reduced-distance between a point and a node"""
    if tree.euclidean:
        return euclidean_dist_to_rdist64(
            max_dist64(tree, i_node, pt)
        )
    else:
        return tree.dist_metric._dist_to_rdist(
            max_dist64(tree, i_node, pt)
        )


cdef inline float64_t min_dist_dual64(
    BinaryTree64 tree1,
    intp_t i_node1,
    BinaryTree64 tree2,
    intp_t i_node2,
) except -1:
    """compute the minimum distance between two nodes"""
    cdef float64_t dist_pt = tree1.dist(&tree2.node_bounds[0, i_node2, 0],
                                      &tree1.node_bounds[0, i_node1, 0],
                                      tree1.data.shape[1])
    return fmax(0, (dist_pt - tree1.node_data[i_node1].radius
                    - tree2.node_data[i_node2].radius))


cdef inline float64_t max_dist_dual64(
    BinaryTree64 tree1,
    intp_t i_node1,
    BinaryTree64 tree2,
    intp_t i_node2,
) except -1:
    """compute the maximum distance between two nodes"""
    cdef float64_t dist_pt = tree1.dist(&tree2.node_bounds[0, i_node2, 0],
                                      &tree1.node_bounds[0, i_node1, 0],
                                      tree1.data.shape[1])
    return (dist_pt + tree1.node_data[i_node1].radius
            + tree2.node_data[i_node2].radius)


cdef inline float64_t min_rdist_dual64(
    BinaryTree64 tree1,
    intp_t i_node1,
    BinaryTree64 tree2,
    intp_t i_node2,
) except -1:
    """compute the minimum reduced distance between two nodes"""
    if tree1.euclidean:
        return euclidean_dist_to_rdist64(
            min_dist_dual64(tree1, i_node1, tree2, i_node2)
        )
    else:
        return tree1.dist_metric._dist_to_rdist(
            min_dist_dual64(tree1, i_node1, tree2, i_node2)
        )


cdef inline float64_t max_rdist_dual64(
    BinaryTree64 tree1,
    intp_t i_node1,
    BinaryTree64 tree2,
    intp_t i_node2,
) except -1:
    """compute the maximum reduced distance between two nodes"""
    if tree1.euclidean:
        return euclidean_dist_to_rdist64(
            max_dist_dual64(tree1, i_node1, tree2, i_node2)
        )
    else:
        return tree1.dist_metric._dist_to_rdist(
            max_dist_dual64(tree1, i_node1, tree2, i_node2)
        )

cdef int allocate_data32(
    BinaryTree32 tree,
    intp_t n_nodes,
    intp_t n_features,
) except -1:
    """Allocate arrays needed for the KD Tree"""
    tree.node_bounds = np.zeros((1, n_nodes, n_features), dtype=np.float32)
    return 0


cdef int init_node32(
    BinaryTree32 tree,
    NodeData_t[::1] node_data,
    intp_t i_node,
    intp_t idx_start,
    intp_t idx_end,
) except -1:
    """Initialize the node for the dataset stored in tree.data"""
    cdef intp_t n_features = tree.data.shape[1]
    cdef intp_t n_points = idx_end - idx_start

    cdef intp_t i, j
    cdef float64_t radius
    cdef const float32_t *this_pt

    cdef intp_t* idx_array = &tree.idx_array[0]
    cdef const float32_t* data = &tree.data[0, 0]
    cdef float32_t* centroid = &tree.node_bounds[0, i_node, 0]

    cdef bint with_sample_weight = tree.sample_weight is not None
    cdef const float32_t* sample_weight
    cdef float64_t sum_weight_node
    if with_sample_weight:
        sample_weight = &tree.sample_weight[0]

    # determine Node centroid
    for j in range(n_features):
        centroid[j] = 0

    if with_sample_weight:
        sum_weight_node = 0
        for i in range(idx_start, idx_end):
            sum_weight_node += sample_weight[idx_array[i]]
            this_pt = data + n_features * idx_array[i]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j] * sample_weight[idx_array[i]]

        for j in range(n_features):
            centroid[j] /= sum_weight_node
    else:
        for i in range(idx_start, idx_end):
            this_pt = data + n_features * idx_array[i]
            for j from 0 <= j < n_features:
                centroid[j] += this_pt[j]

        for j in range(n_features):
            centroid[j] /= n_points

    # determine Node radius
    radius = 0
    for i in range(idx_start, idx_end):
        radius = fmax(radius,
                      tree.rdist(centroid,
                                 data + n_features * idx_array[i],
                                 n_features))

    node_data[i_node].radius = tree.dist_metric._rdist_to_dist(radius)
    node_data[i_node].idx_start = idx_start
    node_data[i_node].idx_end = idx_end
    return 0


cdef inline float64_t min_dist32(
    BinaryTree32 tree,
    intp_t i_node,
    const float32_t* pt,
) except -1 nogil:
    """Compute the minimum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    return fmax(0, dist_pt - tree.node_data[i_node].radius)


cdef inline float64_t max_dist32(
    BinaryTree32 tree,
    intp_t i_node,
    const float32_t* pt,
) except -1:
    """Compute the maximum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    return dist_pt + tree.node_data[i_node].radius


cdef inline int min_max_dist32(
    BinaryTree32 tree,
    intp_t i_node,
    const float32_t* pt,
    float64_t* min_dist,
    float64_t* max_dist,
) except -1 nogil:
    """Compute the minimum and maximum distance between a point and a node"""
    cdef float64_t dist_pt = tree.dist(pt, &tree.node_bounds[0, i_node, 0],
                                     tree.data.shape[1])
    cdef float64_t rad = tree.node_data[i_node].radius
    min_dist[0] = fmax(0, dist_pt - rad)
    max_dist[0] = dist_pt + rad
    return 0


cdef inline float64_t min_rdist32(
    BinaryTree32 tree,
    intp_t i_node,
    const float32_t* pt,
) except -1 nogil:
    """Compute the minimum reduced-distance between a point and a node"""
    if tree.euclidean:
        return euclidean_dist_to_rdist32(
            min_dist32(tree, i_node, pt)
        )
    else:
        return tree.dist_metric._dist_to_rdist(
            min_dist32(tree, i_node, pt)
        )


cdef inline float64_t max_rdist32(
    BinaryTree32 tree,
    intp_t i_node,
    const float32_t* pt,
) except -1:
    """Compute the maximum reduced-distance between a point and a node"""
    if tree.euclidean:
        return euclidean_dist_to_rdist32(
            max_dist32(tree, i_node, pt)
        )
    else:
        return tree.dist_metric._dist_to_rdist(
            max_dist32(tree, i_node, pt)
        )


cdef inline float64_t min_dist_dual32(
    BinaryTree32 tree1,
    intp_t i_node1,
    BinaryTree32 tree2,
    intp_t i_node2,
) except -1:
    """compute the minimum distance between two nodes"""
    cdef float64_t dist_pt = tree1.dist(&tree2.node_bounds[0, i_node2, 0],
                                      &tree1.node_bounds[0, i_node1, 0],
                                      tree1.data.shape[1])
    return fmax(0, (dist_pt - tree1.node_data[i_node1].radius
                    - tree2.node_data[i_node2].radius))


cdef inline float64_t max_dist_dual32(
    BinaryTree32 tree1,
    intp_t i_node1,
    BinaryTree32 tree2,
    intp_t i_node2,
) except -1:
    """compute the maximum distance between two nodes"""
    cdef float64_t dist_pt = tree1.dist(&tree2.node_bounds[0, i_node2, 0],
                                      &tree1.node_bounds[0, i_node1, 0],
                                      tree1.data.shape[1])
    return (dist_pt + tree1.node_data[i_node1].radius
            + tree2.node_data[i_node2].radius)


cdef inline float64_t min_rdist_dual32(
    BinaryTree32 tree1,
    intp_t i_node1,
    BinaryTree32 tree2,
    intp_t i_node2,
) except -1:
    """compute the minimum reduced distance between two nodes"""
    if tree1.euclidean:
        return euclidean_dist_to_rdist32(
            min_dist_dual32(tree1, i_node1, tree2, i_node2)
        )
    else:
        return tree1.dist_metric._dist_to_rdist(
            min_dist_dual32(tree1, i_node1, tree2, i_node2)
        )


cdef inline float64_t max_rdist_dual32(
    BinaryTree32 tree1,
    intp_t i_node1,
    BinaryTree32 tree2,
    intp_t i_node2,
) except -1:
    """compute the maximum reduced distance between two nodes"""
    if tree1.euclidean:
        return euclidean_dist_to_rdist32(
            max_dist_dual32(tree1, i_node1, tree2, i_node2)
        )
    else:
        return tree1.dist_metric._dist_to_rdist(
            max_dist_dual32(tree1, i_node1, tree2, i_node2)
        )


class BallTree(BallTree64):
    __doc__ = CLASS_DOC.format(BinaryTree="BallTree")
    pass
