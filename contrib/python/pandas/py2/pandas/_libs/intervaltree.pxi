"""
Template for intervaltree

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

ctypedef fused scalar_t:
    float64_t
    float32_t
    int64_t
    int32_t
    uint64_t

# ----------------------------------------------------------------------
# IntervalTree
# ----------------------------------------------------------------------

cdef class IntervalTree(IntervalMixin):
    """A centered interval tree

    Based off the algorithm described on Wikipedia:
    http://en.wikipedia.org/wiki/Interval_tree

    we are emulating the IndexEngine interface
    """
    cdef:
        readonly object left, right, root, dtype
        readonly str closed
        object _is_overlapping, _left_sorter, _right_sorter

    def __init__(self, left, right, closed='right', leaf_size=100):
        """
        Parameters
        ----------
        left, right : np.ndarray[ndim=1]
            Left and right bounds for each interval. Assumed to contain no
            NaNs.
        closed : {'left', 'right', 'both', 'neither'}, optional
            Whether the intervals are closed on the left-side, right-side, both
            or neither. Defaults to 'right'.
        leaf_size : int, optional
            Parameter that controls when the tree switches from creating nodes
            to brute-force search. Tune this parameter to optimize query
            performance.
        """
        if closed not in ['left', 'right', 'both', 'neither']:
            raise ValueError("invalid option for 'closed': %s" % closed)

        left = np.asarray(left)
        right = np.asarray(right)
        self.dtype = np.result_type(left, right)
        self.left = np.asarray(left, dtype=self.dtype)
        self.right = np.asarray(right, dtype=self.dtype)

        indices = np.arange(len(left), dtype='int64')

        self.closed = closed

        # GH 23352: ensure no nan in nodes
        mask = ~np.isnan(self.left)
        self.left = self.left[mask]
        self.right = self.right[mask]
        indices = indices[mask]

        node_cls = NODE_CLASSES[str(self.dtype), closed]
        self.root = node_cls(self.left, self.right, indices, leaf_size)

    @property
    def left_sorter(self):
        """How to sort the left labels; this is used for binary search
        """
        if self._left_sorter is None:
            self._left_sorter = np.argsort(self.left)
        return self._left_sorter

    @property
    def right_sorter(self):
        """How to sort the right labels
        """
        if self._right_sorter is None:
            self._right_sorter = np.argsort(self.right)
        return self._right_sorter

    @property
    def is_overlapping(self):
        """
        Determine if the IntervalTree contains overlapping intervals.
        Cached as self._is_overlapping.
        """
        if self._is_overlapping is not None:
            return self._is_overlapping

        # <= when both sides closed since endpoints can overlap
        op = le if self.closed == 'both' else lt

        # overlap if start of current interval < end of previous interval
        # (current and previous in terms of sorted order by left/start side)
        current = self.left[self.left_sorter[1:]]
        previous = self.right[self.left_sorter[:-1]]
        self._is_overlapping = bool(op(current, previous).any())

        return self._is_overlapping

    def get_loc(self, scalar_t key):
        """Return all positions corresponding to intervals that overlap with
        the given scalar key
        """
        result = Int64Vector()
        self.root.query(result, key)
        if not result.data.n:
            raise KeyError(key)
        return result.to_array().astype('intp')

    def _get_partial_overlap(self, key_left, key_right, side):
        """Return all positions corresponding to intervals with the given side
        falling between the left and right bounds of an interval query
        """
        if side == 'left':
            values = self.left
            sorter = self.left_sorter
        else:
            values = self.right
            sorter = self.right_sorter
        key = [key_left, key_right]
        i, j = values.searchsorted(key, sorter=sorter)
        return sorter[i:j]

    def get_loc_interval(self, key_left, key_right):
        """Lookup the intervals enclosed in the given interval bounds

        The given interval is presumed to have closed bounds.
        """
        import pandas as pd
        left_overlap = self._get_partial_overlap(key_left, key_right, 'left')
        right_overlap = self._get_partial_overlap(key_left, key_right, 'right')
        enclosing = self.get_loc(0.5 * (key_left + key_right))
        combined = np.concatenate([left_overlap, right_overlap, enclosing])
        uniques = pd.unique(combined)
        return uniques.astype('intp')

    def get_indexer(self, scalar_t[:] target):
        """Return the positions corresponding to unique intervals that overlap
        with the given array of scalar targets.
        """

        # TODO: write get_indexer_intervals
        cdef:
            size_t old_len
            Py_ssize_t i
            Int64Vector result

        result = Int64Vector()
        old_len = 0
        for i in range(len(target)):
            self.root.query(result, target[i])
            if result.data.n == old_len:
                result.append(-1)
            elif result.data.n > old_len + 1:
                raise KeyError(
                    'indexer does not intersect a unique set of intervals')
            old_len = result.data.n
        return result.to_array().astype('intp')

    def get_indexer_non_unique(self, scalar_t[:] target):
        """Return the positions corresponding to intervals that overlap with
        the given array of scalar targets. Non-unique positions are repeated.
        """
        cdef:
            size_t old_len
            Py_ssize_t i
            Int64Vector result, missing

        result = Int64Vector()
        missing = Int64Vector()
        old_len = 0
        for i in range(len(target)):
            self.root.query(result, target[i])
            if result.data.n == old_len:
                result.append(-1)
                missing.append(i)
            old_len = result.data.n
        return (result.to_array().astype('intp'),
                missing.to_array().astype('intp'))

    def __repr__(self):
        return ('<IntervalTree[{dtype},{closed}]: '
                '{n_elements} elements>'.format(
                    dtype=self.dtype, closed=self.closed,
                    n_elements=self.root.n_elements))

    # compat with IndexEngine interface
    def clear_mapping(self):
        pass


cdef take(ndarray source, ndarray indices):
    """Take the given positions from a 1D ndarray
    """
    return PyArray_Take(source, indices, 0)


cdef sort_values_and_indices(all_values, all_indices, subset):
    indices = take(all_indices, subset)
    values = take(all_values, subset)
    sorter = PyArray_ArgSort(values, 0, NPY_QUICKSORT)
    sorted_values = take(values, sorter)
    sorted_indices = take(indices, sorter)
    return sorted_values, sorted_indices


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------

# we need specialized nodes and leaves to optimize for different dtype and
# closed values

NODE_CLASSES = {}

cdef class Float32ClosedLeftIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float32ClosedLeftIntervalNode left_node, right_node
        float32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float32_t min_left, max_right
        readonly float32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float32_t, ndim=1] left,
                 ndarray[float32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float32_t[:] left, float32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float32_t, ndim=1] left,
                        ndarray[float32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float32ClosedLeftIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float32ClosedLeftIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float32ClosedLeftIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float32',
             'left'] = Float32ClosedLeftIntervalNode

cdef class Float32ClosedRightIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float32ClosedRightIntervalNode left_node, right_node
        float32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float32_t min_left, max_right
        readonly float32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float32_t, ndim=1] left,
                 ndarray[float32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float32_t[:] left, float32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float32_t, ndim=1] left,
                        ndarray[float32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float32ClosedRightIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float32ClosedRightIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float32ClosedRightIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float32',
             'right'] = Float32ClosedRightIntervalNode

cdef class Float32ClosedBothIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float32ClosedBothIntervalNode left_node, right_node
        float32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float32_t min_left, max_right
        readonly float32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float32_t, ndim=1] left,
                 ndarray[float32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float32_t[:] left, float32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float32_t, ndim=1] left,
                        ndarray[float32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float32ClosedBothIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float32ClosedBothIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float32ClosedBothIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float32',
             'both'] = Float32ClosedBothIntervalNode

cdef class Float32ClosedNeitherIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float32ClosedNeitherIntervalNode left_node, right_node
        float32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float32_t min_left, max_right
        readonly float32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float32_t, ndim=1] left,
                 ndarray[float32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float32_t[:] left, float32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float32_t, ndim=1] left,
                        ndarray[float32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float32ClosedNeitherIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float32ClosedNeitherIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float32ClosedNeitherIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float32',
             'neither'] = Float32ClosedNeitherIntervalNode

cdef class Float64ClosedLeftIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float64ClosedLeftIntervalNode left_node, right_node
        float64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float64_t min_left, max_right
        readonly float64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float64_t, ndim=1] left,
                 ndarray[float64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float64_t[:] left, float64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float64_t, ndim=1] left,
                        ndarray[float64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float64ClosedLeftIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float64ClosedLeftIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float64ClosedLeftIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float64',
             'left'] = Float64ClosedLeftIntervalNode

cdef class Float64ClosedRightIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float64ClosedRightIntervalNode left_node, right_node
        float64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float64_t min_left, max_right
        readonly float64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float64_t, ndim=1] left,
                 ndarray[float64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float64_t[:] left, float64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float64_t, ndim=1] left,
                        ndarray[float64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float64ClosedRightIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float64ClosedRightIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float64ClosedRightIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float64',
             'right'] = Float64ClosedRightIntervalNode

cdef class Float64ClosedBothIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float64ClosedBothIntervalNode left_node, right_node
        float64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float64_t min_left, max_right
        readonly float64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float64_t, ndim=1] left,
                 ndarray[float64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float64_t[:] left, float64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float64_t, ndim=1] left,
                        ndarray[float64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float64ClosedBothIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float64ClosedBothIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float64ClosedBothIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float64',
             'both'] = Float64ClosedBothIntervalNode

cdef class Float64ClosedNeitherIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Float64ClosedNeitherIntervalNode left_node, right_node
        float64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        float64_t min_left, max_right
        readonly float64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[float64_t, ndim=1] left,
                 ndarray[float64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, float64_t[:] left, float64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[float64_t, ndim=1] left,
                        ndarray[float64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Float64ClosedNeitherIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            float64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Float64ClosedNeitherIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Float64ClosedNeitherIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['float64',
             'neither'] = Float64ClosedNeitherIntervalNode

cdef class Int32ClosedLeftIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int32ClosedLeftIntervalNode left_node, right_node
        int32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int32_t min_left, max_right
        readonly int32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int32_t, ndim=1] left,
                 ndarray[int32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int32_t[:] left, int32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int32_t, ndim=1] left,
                        ndarray[int32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int32ClosedLeftIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int32ClosedLeftIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int32ClosedLeftIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int32',
             'left'] = Int32ClosedLeftIntervalNode

cdef class Int32ClosedRightIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int32ClosedRightIntervalNode left_node, right_node
        int32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int32_t min_left, max_right
        readonly int32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int32_t, ndim=1] left,
                 ndarray[int32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int32_t[:] left, int32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int32_t, ndim=1] left,
                        ndarray[int32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int32ClosedRightIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int32ClosedRightIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int32ClosedRightIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int32',
             'right'] = Int32ClosedRightIntervalNode

cdef class Int32ClosedBothIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int32ClosedBothIntervalNode left_node, right_node
        int32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int32_t min_left, max_right
        readonly int32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int32_t, ndim=1] left,
                 ndarray[int32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int32_t[:] left, int32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int32_t, ndim=1] left,
                        ndarray[int32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int32ClosedBothIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int32ClosedBothIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int32ClosedBothIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int32',
             'both'] = Int32ClosedBothIntervalNode

cdef class Int32ClosedNeitherIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int32ClosedNeitherIntervalNode left_node, right_node
        int32_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int32_t min_left, max_right
        readonly int32_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int32_t, ndim=1] left,
                 ndarray[int32_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int32_t[:] left, int32_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int32_t, ndim=1] left,
                        ndarray[int32_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int32ClosedNeitherIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int32_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int32ClosedNeitherIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int32ClosedNeitherIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int32',
             'neither'] = Int32ClosedNeitherIntervalNode

cdef class Int64ClosedLeftIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int64ClosedLeftIntervalNode left_node, right_node
        int64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int64_t min_left, max_right
        readonly int64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int64_t, ndim=1] left,
                 ndarray[int64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int64_t[:] left, int64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int64_t, ndim=1] left,
                        ndarray[int64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int64ClosedLeftIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int64ClosedLeftIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int64ClosedLeftIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int64',
             'left'] = Int64ClosedLeftIntervalNode

cdef class Int64ClosedRightIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int64ClosedRightIntervalNode left_node, right_node
        int64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int64_t min_left, max_right
        readonly int64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int64_t, ndim=1] left,
                 ndarray[int64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int64_t[:] left, int64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int64_t, ndim=1] left,
                        ndarray[int64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int64ClosedRightIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int64ClosedRightIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int64ClosedRightIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int64',
             'right'] = Int64ClosedRightIntervalNode

cdef class Int64ClosedBothIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int64ClosedBothIntervalNode left_node, right_node
        int64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int64_t min_left, max_right
        readonly int64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int64_t, ndim=1] left,
                 ndarray[int64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int64_t[:] left, int64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int64_t, ndim=1] left,
                        ndarray[int64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int64ClosedBothIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int64ClosedBothIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int64ClosedBothIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int64',
             'both'] = Int64ClosedBothIntervalNode

cdef class Int64ClosedNeitherIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Int64ClosedNeitherIntervalNode left_node, right_node
        int64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        int64_t min_left, max_right
        readonly int64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[int64_t, ndim=1] left,
                 ndarray[int64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, int64_t[:] left, int64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[int64_t, ndim=1] left,
                        ndarray[int64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Int64ClosedNeitherIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            int64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Int64ClosedNeitherIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Int64ClosedNeitherIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['int64',
             'neither'] = Int64ClosedNeitherIntervalNode

cdef class Uint64ClosedLeftIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Uint64ClosedLeftIntervalNode left_node, right_node
        uint64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        uint64_t min_left, max_right
        readonly uint64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[uint64_t, ndim=1] left,
                 ndarray[uint64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, uint64_t[:] left, uint64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[uint64_t, ndim=1] left,
                        ndarray[uint64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Uint64ClosedLeftIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            uint64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Uint64ClosedLeftIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Uint64ClosedLeftIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['uint64',
             'left'] = Uint64ClosedLeftIntervalNode

cdef class Uint64ClosedRightIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Uint64ClosedRightIntervalNode left_node, right_node
        uint64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        uint64_t min_left, max_right
        readonly uint64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[uint64_t, ndim=1] left,
                 ndarray[uint64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, uint64_t[:] left, uint64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[uint64_t, ndim=1] left,
                        ndarray[uint64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Uint64ClosedRightIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            uint64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Uint64ClosedRightIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Uint64ClosedRightIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['uint64',
             'right'] = Uint64ClosedRightIntervalNode

cdef class Uint64ClosedBothIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Uint64ClosedBothIntervalNode left_node, right_node
        uint64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        uint64_t min_left, max_right
        readonly uint64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[uint64_t, ndim=1] left,
                 ndarray[uint64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, uint64_t[:] left, uint64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] < self.pivot:
                left_ind.append(i)
            elif self.pivot < left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[uint64_t, ndim=1] left,
                        ndarray[uint64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Uint64ClosedBothIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            uint64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] <= point <= self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] <= point:
                        break
                    result.append(indices[i])
                if point <= self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point <= values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left <= point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Uint64ClosedBothIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Uint64ClosedBothIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['uint64',
             'both'] = Uint64ClosedBothIntervalNode

cdef class Uint64ClosedNeitherIntervalNode:
    """Non-terminal node for an IntervalTree

    Categorizes intervals by those that fall to the left, those that fall to
    the right, and those that overlap with the pivot.
    """
    cdef:
        Uint64ClosedNeitherIntervalNode left_node, right_node
        uint64_t[:] center_left_values, center_right_values, left, right
        int64_t[:] center_left_indices, center_right_indices, indices
        uint64_t min_left, max_right
        readonly uint64_t pivot
        readonly int64_t n_elements, n_center, leaf_size
        readonly bint is_leaf_node

    def __init__(self,
                 ndarray[uint64_t, ndim=1] left,
                 ndarray[uint64_t, ndim=1] right,
                 ndarray[int64_t, ndim=1] indices,
                 int64_t leaf_size):

        self.n_elements = len(left)
        self.leaf_size = leaf_size

        # min_left and min_right are used to speed-up query by skipping
        # query on sub-nodes. If this node has size 0, query is cheap,
        # so these values don't matter.
        if left.size > 0:
            self.min_left = left.min()
            self.max_right = right.max()
        else:
            self.min_left = 0
            self.max_right = 0

        if self.n_elements <= leaf_size:
            # make this a terminal (leaf) node
            self.is_leaf_node = True
            self.left = left
            self.right = right
            self.indices = indices
            self.n_center = 0
        else:
            # calculate a pivot so we can create child nodes
            self.is_leaf_node = False
            self.pivot = np.median(left / 2 + right / 2)
            left_set, right_set, center_set = self.classify_intervals(
                left, right)

            self.left_node = self.new_child_node(left, right,
                                                 indices, left_set)
            self.right_node = self.new_child_node(left, right,
                                                  indices, right_set)

            self.center_left_values, self.center_left_indices = \
                sort_values_and_indices(left, indices, center_set)
            self.center_right_values, self.center_right_indices = \
                sort_values_and_indices(right, indices, center_set)
            self.n_center = len(self.center_left_indices)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef classify_intervals(self, uint64_t[:] left, uint64_t[:] right):
        """Classify the given intervals based upon whether they fall to the
        left, right, or overlap with this node's pivot.
        """
        cdef:
            Int64Vector left_ind, right_ind, overlapping_ind
            Py_ssize_t i

        left_ind = Int64Vector()
        right_ind = Int64Vector()
        overlapping_ind = Int64Vector()

        for i in range(self.n_elements):
            if right[i] <= self.pivot:
                left_ind.append(i)
            elif self.pivot <= left[i]:
                right_ind.append(i)
            else:
                overlapping_ind.append(i)

        return (left_ind.to_array(),
                right_ind.to_array(),
                overlapping_ind.to_array())

    cdef new_child_node(self,
                        ndarray[uint64_t, ndim=1] left,
                        ndarray[uint64_t, ndim=1] right,
                        ndarray[int64_t, ndim=1] indices,
                        ndarray[int64_t, ndim=1] subset):
        """Create a new child node.
        """
        left = take(left, subset)
        right = take(right, subset)
        indices = take(indices, subset)
        return Uint64ClosedNeitherIntervalNode(
            left, right, indices, self.leaf_size)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef query(self, Int64Vector result, scalar_t point):
        """Recursively query this node and its sub-nodes for intervals that
        overlap with the query point.
        """
        cdef:
            int64_t[:] indices
            uint64_t[:] values
            Py_ssize_t i

        if self.is_leaf_node:
            # Once we get down to a certain size, it doesn't make sense to
            # continue the binary tree structure. Instead, we use linear
            # search.
            for i in range(self.n_elements):
                if self.left[i] < point < self.right[i]:
                    result.append(self.indices[i])
        else:
            # There are child nodes. Based on comparing our query to the pivot,
            # look at the center values, then go to the relevant child.
            if point < self.pivot:
                values = self.center_left_values
                indices = self.center_left_indices
                for i in range(self.n_center):
                    if not values[i] < point:
                        break
                    result.append(indices[i])
                if point < self.left_node.max_right:
                    self.left_node.query(result, point)
            elif point > self.pivot:
                values = self.center_right_values
                indices = self.center_right_indices
                for i in range(self.n_center - 1, -1, -1):
                    if not point < values[i]:
                        break
                    result.append(indices[i])
                if self.right_node.min_left < point:
                    self.right_node.query(result, point)
            else:
                result.extend(self.center_left_indices)

    def __repr__(self):
        if self.is_leaf_node:
            return ('<Uint64ClosedNeitherIntervalNode: '
                    '%s elements (terminal)>' % self.n_elements)
        else:
            n_left = self.left_node.n_elements
            n_right = self.right_node.n_elements
            n_center = self.n_elements - n_left - n_right
            return ('<Uint64ClosedNeitherIntervalNode: '
                    'pivot %s, %s elements (%s left, %s right, %s '
                    'overlapping)>' % (self.pivot, self.n_elements,
                                       n_left, n_right, n_center))

    def counts(self):
        """
        Inspect counts on this node
        useful for debugging purposes
        """
        if self.is_leaf_node:
            return self.n_elements
        else:
            m = len(self.center_left_values)
            l = self.left_node.counts()
            r = self.right_node.counts()
            return (m, (l, r))

NODE_CLASSES['uint64',
             'neither'] = Uint64ClosedNeitherIntervalNode
