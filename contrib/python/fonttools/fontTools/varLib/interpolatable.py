"""
Tool to find wrong contour order between different masters, and
other interpolatability (or lack thereof) issues.

Call as:
$ fonttools varLib.interpolatable font1 font2 ...
"""

from fontTools.pens.basePen import AbstractPen, BasePen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.statisticsPen import StatisticsPen, StatisticsControlPen
from fontTools.pens.momentsPen import OpenContourError
from fontTools.varLib.models import piecewiseLinearMap, normalizeLocation
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from functools import wraps
from pprint import pformat
from math import sqrt, copysign, atan2, pi
import itertools
import logging

log = logging.getLogger("fontTools.varLib.interpolatable")


def _rot_list(l, k):
    """Rotate list by k items forward.  Ie. item at position 0 will be
    at position k in returned list.  Negative k is allowed."""
    return l[-k:] + l[:-k]


class PerContourPen(BasePen):
    def __init__(self, Pen, glyphset=None):
        BasePen.__init__(self, glyphset)
        self._glyphset = glyphset
        self._Pen = Pen
        self._pen = None
        self.value = []

    def _moveTo(self, p0):
        self._newItem()
        self._pen.moveTo(p0)

    def _lineTo(self, p1):
        self._pen.lineTo(p1)

    def _qCurveToOne(self, p1, p2):
        self._pen.qCurveTo(p1, p2)

    def _curveToOne(self, p1, p2, p3):
        self._pen.curveTo(p1, p2, p3)

    def _closePath(self):
        self._pen.closePath()
        self._pen = None

    def _endPath(self):
        self._pen.endPath()
        self._pen = None

    def _newItem(self):
        self._pen = pen = self._Pen()
        self.value.append(pen)


class PerContourOrComponentPen(PerContourPen):
    def addComponent(self, glyphName, transformation):
        self._newItem()
        self.value[-1].addComponent(glyphName, transformation)


class SimpleRecordingPointPen(AbstractPointPen):
    def __init__(self):
        self.value = []

    def beginPath(self, identifier=None, **kwargs):
        pass

    def endPath(self) -> None:
        pass

    def addPoint(self, pt, segmentType=None):
        self.value.append((pt, False if segmentType is None else True))


def _vdiff_hypot2(v0, v1):
    s = 0
    for x0, x1 in zip(v0, v1):
        d = x1 - x0
        s += d * d
    return s


def _vdiff_hypot2_complex(v0, v1):
    s = 0
    for x0, x1 in zip(v0, v1):
        d = x1 - x0
        s += d.real * d.real + d.imag * d.imag
        # This does the same but seems to be slower:
        # s += (d * d.conjugate()).real
    return s


def _hypot2_complex(d):
    return d.real * d.real + d.imag * d.imag


def _matching_cost(G, matching):
    return sum(G[i][j] for i, j in enumerate(matching))


def min_cost_perfect_bipartite_matching_scipy(G):
    n = len(G)
    rows, cols = linear_sum_assignment(G)
    assert (rows == list(range(n))).all()
    return list(cols), _matching_cost(G, cols)


def min_cost_perfect_bipartite_matching_munkres(G):
    n = len(G)
    cols = [None] * n
    for row, col in Munkres().compute(G):
        cols[row] = col
    return cols, _matching_cost(G, cols)


def min_cost_perfect_bipartite_matching_bruteforce(G):
    n = len(G)

    if n > 6:
        raise Exception("Install Python module 'munkres' or 'scipy >= 0.17.0'")

    # Otherwise just brute-force
    permutations = itertools.permutations(range(n))
    best = list(next(permutations))
    best_cost = _matching_cost(G, best)
    for p in permutations:
        cost = _matching_cost(G, p)
        if cost < best_cost:
            best, best_cost = list(p), cost
    return best, best_cost


try:
    from scipy.optimize import linear_sum_assignment

    min_cost_perfect_bipartite_matching = min_cost_perfect_bipartite_matching_scipy
except ImportError:
    try:
        from munkres import Munkres

        min_cost_perfect_bipartite_matching = (
            min_cost_perfect_bipartite_matching_munkres
        )
    except ImportError:
        min_cost_perfect_bipartite_matching = (
            min_cost_perfect_bipartite_matching_bruteforce
        )


def _contour_vector_from_stats(stats):
    # Don't change the order of items here.
    # It's okay to add to the end, but otherwise, other
    # code depends on it. Search for "covariance".
    size = sqrt(abs(stats.area))
    return (
        copysign((size), stats.area),
        stats.meanX,
        stats.meanY,
        stats.stddevX * 2,
        stats.stddevY * 2,
        stats.correlation * size,
    )


def _points_characteristic_bits(points):
    bits = 0
    for pt, b in reversed(points):
        bits = (bits << 1) | b
    return bits


_NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR = 4


def _points_complex_vector(points):
    vector = []
    if not points:
        return vector
    points = [complex(*pt) for pt, _ in points]
    n = len(points)
    assert _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR == 4
    points.extend(points[: _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR - 1])
    while len(points) < _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR:
        points.extend(points[: _NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR - 1])
    for i in range(n):
        # The weights are magic numbers.

        # The point itself
        p0 = points[i]
        vector.append(p0)

        # The vector to the next point
        p1 = points[i + 1]
        d0 = p1 - p0
        vector.append(d0 * 3)

        # The turn vector
        p2 = points[i + 2]
        d1 = p2 - p1
        vector.append(d1 - d0)

        # The angle to the next point, as a cross product;
        # Square root of, to match dimentionality of distance.
        cross = d0.real * d1.imag - d0.imag * d1.real
        cross = copysign(sqrt(abs(cross)), cross)
        vector.append(cross * 4)

    return vector


def _add_isomorphisms(points, isomorphisms, reverse):
    reference_bits = _points_characteristic_bits(points)
    n = len(points)

    # if points[0][0] == points[-1][0]:
    #   abort

    if reverse:
        points = points[::-1]
        bits = _points_characteristic_bits(points)
    else:
        bits = reference_bits

    vector = _points_complex_vector(points)

    assert len(vector) % n == 0
    mult = len(vector) // n
    mask = (1 << n) - 1

    for i in range(n):
        b = ((bits << (n - i)) & mask) | (bits >> i)
        if b == reference_bits:
            isomorphisms.append(
                (_rot_list(vector, -i * mult), n - 1 - i if reverse else i, reverse)
            )


def _find_parents_and_order(glyphsets, locations):
    parents = [None] + list(range(len(glyphsets) - 1))
    order = list(range(len(glyphsets)))
    if locations:
        # Order base master first
        bases = (i for i, l in enumerate(locations) if all(v == 0 for v in l.values()))
        if bases:
            base = next(bases)
            logging.info("Base master index %s, location %s", base, locations[base])
        else:
            base = 0
            logging.warning("No base master location found")

        # Form a minimum spanning tree of the locations
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree

            graph = [[0] * len(locations) for _ in range(len(locations))]
            axes = set()
            for l in locations:
                axes.update(l.keys())
            axes = sorted(axes)
            vectors = [tuple(l.get(k, 0) for k in axes) for l in locations]
            for i, j in itertools.combinations(range(len(locations)), 2):
                graph[i][j] = _vdiff_hypot2(vectors[i], vectors[j])

            tree = minimum_spanning_tree(graph)
            rows, cols = tree.nonzero()
            graph = defaultdict(set)
            for row, col in zip(rows, cols):
                graph[row].add(col)
                graph[col].add(row)

            # Traverse graph from the base and assign parents
            parents = [None] * len(locations)
            order = []
            visited = set()
            queue = deque([base])
            while queue:
                i = queue.popleft()
                visited.add(i)
                order.append(i)
                for j in sorted(graph[i]):
                    if j not in visited:
                        parents[j] = i
                        queue.append(j)

        except ImportError:
            pass

        log.info("Parents: %s", parents)
        log.info("Order: %s", order)
    return parents, order


def test_gen(
    glyphsets,
    glyphs=None,
    names=None,
    ignore_missing=False,
    *,
    locations=None,
    tolerance=0.95,
    show_all=False,
):
    if names is None:
        names = glyphsets

    if glyphs is None:
        # `glyphs = glyphsets[0].keys()` is faster, certainly, but doesn't allow for sparse TTFs/OTFs given out of order
        # ... risks the sparse master being the first one, and only processing a subset of the glyphs
        glyphs = {g for glyphset in glyphsets for g in glyphset.keys()}

    parents, order = _find_parents_and_order(glyphsets, locations)

    def grand_parent(i, glyphname):
        if i is None:
            return None
        i = parents[i]
        if i is None:
            return None
        while parents[i] is not None and glyphsets[i][glyphname] is None:
            i = parents[i]
        return i

    for glyph_name in glyphs:
        log.info("Testing glyph %s", glyph_name)
        allGreenVectors = []
        allControlVectors = []
        allNodeTypes = []
        allContourIsomorphisms = []
        allContourPoints = []
        allGlyphs = [glyphset[glyph_name] for glyphset in glyphsets]
        if len([1 for glyph in allGlyphs if glyph is not None]) <= 1:
            continue
        for master_idx, (glyph, glyphset, name) in enumerate(
            zip(allGlyphs, glyphsets, names)
        ):
            if glyph is None:
                if not ignore_missing:
                    yield (
                        glyph_name,
                        {"type": "missing", "master": name, "master_idx": master_idx},
                    )
                allNodeTypes.append(None)
                allControlVectors.append(None)
                allGreenVectors.append(None)
                allContourIsomorphisms.append(None)
                allContourPoints.append(None)
                continue

            perContourPen = PerContourOrComponentPen(RecordingPen, glyphset=glyphset)
            try:
                glyph.draw(perContourPen, outputImpliedClosingLine=True)
            except TypeError:
                glyph.draw(perContourPen)
            contourPens = perContourPen.value
            del perContourPen

            contourControlVectors = []
            contourGreenVectors = []
            contourIsomorphisms = []
            contourPoints = []
            nodeTypes = []
            allNodeTypes.append(nodeTypes)
            allControlVectors.append(contourControlVectors)
            allGreenVectors.append(contourGreenVectors)
            allContourIsomorphisms.append(contourIsomorphisms)
            allContourPoints.append(contourPoints)
            for ix, contour in enumerate(contourPens):
                contourOps = tuple(op for op, arg in contour.value)
                nodeTypes.append(contourOps)

                greenStats = StatisticsPen(glyphset=glyphset)
                controlStats = StatisticsControlPen(glyphset=glyphset)
                try:
                    contour.replay(greenStats)
                    contour.replay(controlStats)
                except OpenContourError as e:
                    yield (
                        glyph_name,
                        {
                            "master": name,
                            "master_idx": master_idx,
                            "contour": ix,
                            "type": "open_path",
                        },
                    )
                    continue
                contourGreenVectors.append(_contour_vector_from_stats(greenStats))
                contourControlVectors.append(_contour_vector_from_stats(controlStats))

                # Check starting point
                if contourOps[0] == "addComponent":
                    continue
                assert contourOps[0] == "moveTo"
                assert contourOps[-1] in ("closePath", "endPath")
                points = SimpleRecordingPointPen()
                converter = SegmentToPointPen(points, False)
                contour.replay(converter)
                # points.value is a list of pt,bool where bool is true if on-curve and false if off-curve;
                # now check all rotations and mirror-rotations of the contour and build list of isomorphic
                # possible starting points.

                isomorphisms = []
                contourIsomorphisms.append(isomorphisms)

                # Add rotations
                _add_isomorphisms(points.value, isomorphisms, False)
                # Add mirrored rotations
                _add_isomorphisms(points.value, isomorphisms, True)

                contourPoints.append(points.value)

        matchings = [None] * len(allControlVectors)

        for m1idx in order:
            if allNodeTypes[m1idx] is None:
                continue
            m0idx = grand_parent(m1idx, glyph_name)
            if m0idx is None:
                continue
            if allNodeTypes[m0idx] is None:
                continue

            showed = False

            m1 = allNodeTypes[m1idx]
            m0 = allNodeTypes[m0idx]
            if len(m0) != len(m1):
                showed = True
                yield (
                    glyph_name,
                    {
                        "type": "path_count",
                        "master_1": names[m0idx],
                        "master_2": names[m1idx],
                        "master_1_idx": m0idx,
                        "master_2_idx": m1idx,
                        "value_1": len(m0),
                        "value_2": len(m1),
                    },
                )
                continue

            if m0 != m1:
                for pathIx, (nodes1, nodes2) in enumerate(zip(m0, m1)):
                    if nodes1 == nodes2:
                        continue
                    if len(nodes1) != len(nodes2):
                        showed = True
                        yield (
                            glyph_name,
                            {
                                "type": "node_count",
                                "path": pathIx,
                                "master_1": names[m0idx],
                                "master_2": names[m1idx],
                                "master_1_idx": m0idx,
                                "master_2_idx": m1idx,
                                "value_1": len(nodes1),
                                "value_2": len(nodes2),
                            },
                        )
                        continue
                    for nodeIx, (n1, n2) in enumerate(zip(nodes1, nodes2)):
                        if n1 != n2:
                            showed = True
                            yield (
                                glyph_name,
                                {
                                    "type": "node_incompatibility",
                                    "path": pathIx,
                                    "node": nodeIx,
                                    "master_1": names[m0idx],
                                    "master_2": names[m1idx],
                                    "master_1_idx": m0idx,
                                    "master_2_idx": m1idx,
                                    "value_1": n1,
                                    "value_2": n2,
                                },
                            )
                            continue

            m1Control = allControlVectors[m1idx]
            m1Green = allGreenVectors[m1idx]
            m0Control = allControlVectors[m0idx]
            m0Green = allGreenVectors[m0idx]
            if len(m1Control) > 1:
                identity_matching = list(range(len(m0Control)))

                # We try matching both the StatisticsControlPen vector
                # and the StatisticsPen vector.
                # If either method found a identity matching, accept it.
                # This is crucial for fonts like Kablammo[MORF].ttf and
                # Nabla[EDPT,EHLT].ttf, since they really confuse the
                # StatisticsPen vector because of their area=0 contours.
                #
                # TODO: Optimize by only computing the StatisticsPen vector
                # and then checking if it is the identity vector. Only if
                # not, compute the StatisticsControlPen vector and check both.

                costsControl = [
                    [_vdiff_hypot2(v0, v1) for v1 in m1Control] for v0 in m0Control
                ]
                (
                    matching_control,
                    matching_cost_control,
                ) = min_cost_perfect_bipartite_matching(costsControl)
                identity_cost_control = sum(
                    costsControl[i][i] for i in range(len(m0Control))
                )
                done = matching_cost_control == identity_cost_control

                if not done:
                    costsGreen = [
                        [_vdiff_hypot2(v0, v1) for v1 in m1Green] for v0 in m0Green
                    ]
                    (
                        matching_green,
                        matching_cost_green,
                    ) = min_cost_perfect_bipartite_matching(costsGreen)
                    identity_cost_green = sum(
                        costsGreen[i][i] for i in range(len(m0Control))
                    )
                    done = matching_cost_green == identity_cost_green

                if not done:
                    # Otherwise, use the worst of the two matchings.
                    if (
                        matching_cost_control / identity_cost_control
                        < matching_cost_green / identity_cost_green
                    ):
                        matching = matching_control
                        matching_cost = matching_cost_control
                        identity_cost = identity_cost_control
                    else:
                        matching = matching_green
                        matching_cost = matching_cost_green
                        identity_cost = identity_cost_green

                    if matching_cost < identity_cost * tolerance:
                        # print(matching_cost_control / identity_cost_control, matching_cost_green / identity_cost_green)

                        showed = True
                        yield (
                            glyph_name,
                            {
                                "type": "contour_order",
                                "master_1": names[m0idx],
                                "master_2": names[m1idx],
                                "master_1_idx": m0idx,
                                "master_2_idx": m1idx,
                                "value_1": list(range(len(m0Control))),
                                "value_2": matching,
                            },
                        )
                        matchings[m1idx] = matching

            m1 = allContourIsomorphisms[m1idx]
            m0 = allContourIsomorphisms[m0idx]

            # If contour-order is wrong, adjust it
            if matchings[m1idx] is not None and m1:  # m1 is empty for composite glyphs
                m1 = [m1[i] for i in matchings[m1idx]]

            for ix, (contour0, contour1) in enumerate(zip(m0, m1)):
                if len(contour0) == 0 or len(contour0) != len(contour1):
                    # We already reported this; or nothing to do; or not compatible
                    # after reordering above.
                    continue

                c0 = contour0[0]
                # Next few lines duplicated below.
                costs = [_vdiff_hypot2_complex(c0[0], c1[0]) for c1 in contour1]
                min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
                first_cost = costs[0]

                if min_cost < first_cost * tolerance:
                    # c0 is the first isomorphism of the m0 master
                    # contour1 is list of all isomorphisms of the m1 master
                    #
                    # If the two shapes are both circle-ish and slightly
                    # rotated, we detect wrong start point. This is for
                    # example the case hundreds of times in
                    # RobotoSerif-Italic[GRAD,opsz,wdth,wght].ttf
                    #
                    # If the proposed point is only one off from the first
                    # point (and not reversed), try harder:
                    #
                    # Find the major eigenvector of the covariance matrix,
                    # and rotate the contours by that angle. Then find the
                    # closest point again.  If it matches this time, let it
                    # pass.

                    proposed_point = contour1[min_cost_idx][1]
                    reverse = contour1[min_cost_idx][2]
                    num_points = len(allContourPoints[m1idx][ix])
                    leeway = 3
                    okay = False
                    if not reverse and (
                        proposed_point <= leeway
                        or proposed_point >= num_points - leeway
                    ):
                        # Try harder

                        m0Vectors = allGreenVectors[m1idx][ix]
                        m1Vectors = allGreenVectors[m1idx][ix]

                        # Recover the covariance matrix from the GreenVectors.
                        # This is a 2x2 matrix.
                        transforms = []
                        for vector in (m0Vectors, m1Vectors):
                            meanX = vector[1]
                            meanY = vector[2]
                            stddevX = vector[3] / 2
                            stddevY = vector[4] / 2
                            correlation = vector[5] / abs(vector[0])

                            # https://cookierobotics.com/007/
                            a = stddevX * stddevX  # VarianceX
                            c = stddevY * stddevY  # VarianceY
                            b = correlation * stddevX * stddevY  # Covariance

                            delta = (((a - c) * 0.5) ** 2 + b * b) ** 0.5
                            lambda1 = (a + c) * 0.5 + delta  # Major eigenvalue
                            lambda2 = (a + c) * 0.5 - delta  # Minor eigenvalue
                            theta = (
                                atan2(lambda1 - a, b)
                                if b != 0
                                else (pi * 0.5 if a < c else 0)
                            )
                            trans = Transform()
                            trans = trans.translate(meanX, meanY)
                            trans = trans.rotate(theta)
                            trans = trans.scale(sqrt(lambda1), sqrt(lambda2))
                            transforms.append(trans)

                        trans = transforms[0]
                        new_c0 = (
                            [
                                complex(*trans.transformPoint((pt.real, pt.imag)))
                                for pt in c0[0]
                            ],
                        ) + c0[1:]
                        trans = transforms[1]
                        new_contour1 = []
                        for c1 in contour1:
                            new_c1 = (
                                [
                                    complex(*trans.transformPoint((pt.real, pt.imag)))
                                    for pt in c1[0]
                                ],
                            ) + c1[1:]
                            new_contour1.append(new_c1)

                        # Next few lines duplicate from above.
                        costs = [
                            _vdiff_hypot2_complex(new_c0[0], new_c1[0])
                            for new_c1 in new_contour1
                        ]
                        min_cost_idx, min_cost = min(
                            enumerate(costs), key=lambda x: x[1]
                        )
                        first_cost = costs[0]
                        # Only accept a perfect match
                        if min_cost_idx == 0:
                            okay = True

                    if not okay:
                        showed = True
                        yield (
                            glyph_name,
                            {
                                "type": "wrong_start_point",
                                "contour": ix,
                                "master_1": names[m0idx],
                                "master_2": names[m1idx],
                                "master_1_idx": m0idx,
                                "master_2_idx": m1idx,
                                "value_1": 0,
                                "value_2": proposed_point,
                                "reversed": reverse,
                            },
                        )
                else:
                    # If first_cost is Too Large™, do further inspection.
                    # This can happen specially in the case of TrueType
                    # fonts, where the original contour had wrong start point,
                    # but because of the cubic->quadratic conversion, we don't
                    # have many isomorphisms to work with.

                    # The threshold here is all black magic. It's just here to
                    # speed things up so we don't end up doing a full matching
                    # on every contour that is correct.
                    threshold = (
                        len(c0[0]) * (allControlVectors[m0idx][ix][0] * 0.5) ** 2 / 4
                    )  # Magic only
                    c1 = contour1[min_cost_idx]

                    # If point counts are different it's because of the contour
                    # reordering above. We can in theory still try, but our
                    # bipartite-matching implementations currently assume
                    # equal number of vertices on both sides. I'm lazy to update
                    # all three different implementations!

                    if len(c0[0]) == len(c1[0]) and first_cost > threshold:
                        # Do a quick(!) matching between the points. If it's way off,
                        # flag it. This can happen specially in the case of TrueType
                        # fonts, where the original contour had wrong start point, but
                        # because of the cubic->quadratic conversion, we don't have many
                        # isomorphisms.
                        points0 = c0[0][::_NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR]
                        points1 = c1[0][::_NUM_ITEMS_PER_POINTS_COMPLEX_VECTOR]

                        graph = [
                            [_hypot2_complex(p0 - p1) for p1 in points1]
                            for p0 in points0
                        ]
                        matching, matching_cost = min_cost_perfect_bipartite_matching(
                            graph
                        )
                        identity_cost = sum(graph[i][i] for i in range(len(graph)))

                        if matching_cost < identity_cost / 8:  # Heuristic
                            # print(matching_cost, identity_cost, matching)
                            showed = True
                            yield (
                                glyph_name,
                                {
                                    "type": "wrong_structure",
                                    "contour": ix,
                                    "master_1": names[m0idx],
                                    "master_2": names[m1idx],
                                    "master_1_idx": m0idx,
                                    "master_2_idx": m1idx,
                                },
                            )

            if show_all and not showed:
                yield (
                    glyph_name,
                    {
                        "type": "nothing",
                        "master_1": names[m0idx],
                        "master_2": names[m1idx],
                        "master_1_idx": m0idx,
                        "master_2_idx": m1idx,
                    },
                )


@wraps(test_gen)
def test(*args, **kwargs):
    problems = defaultdict(list)
    for glyphname, problem in test_gen(*args, **kwargs):
        problems[glyphname].append(problem)
    return problems


def recursivelyAddGlyph(glyphname, glyphset, ttGlyphSet, glyf):
    if glyphname in glyphset:
        return
    glyphset[glyphname] = ttGlyphSet[glyphname]

    for component in getattr(glyf[glyphname], "components", []):
        recursivelyAddGlyph(component.glyphName, glyphset, ttGlyphSet, glyf)


def main(args=None):
    """Test for interpolatability issues between fonts"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        "fonttools varLib.interpolatable",
        description=main.__doc__,
    )
    parser.add_argument(
        "--glyphs",
        action="store",
        help="Space-separate name of glyphs to check",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all glyph pairs, even if no problems are found",
    )
    parser.add_argument(
        "--tolerance",
        action="store",
        type=float,
        help="Error tolerance. Default 0.95",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format",
    )
    parser.add_argument(
        "--pdf",
        action="store",
        help="Output report in PDF format",
    )
    parser.add_argument(
        "--html",
        action="store",
        help="Output report in HTML format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only exit with code 1 or 0, no output",
    )
    parser.add_argument(
        "--output",
        action="store",
        help="Output file for the problem report; Default: stdout",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Will not report glyphs missing from sparse masters as errors",
    )
    parser.add_argument(
        "inputs",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Input a single variable font / DesignSpace / Glyphs file, or multiple TTF/UFO files",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        type=str,
        action="append",
        help="Name of the master to use in the report. If not provided, all are used.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Run verbosely.")

    args = parser.parse_args(args)

    from fontTools import configLogger

    configLogger(level=("INFO" if args.verbose else "ERROR"))

    glyphs = args.glyphs.split() if args.glyphs else None

    from os.path import basename

    fonts = []
    names = []
    locations = []

    original_args_inputs = tuple(args.inputs)

    if len(args.inputs) == 1:
        designspace = None
        if args.inputs[0].endswith(".designspace"):
            from fontTools.designspaceLib import DesignSpaceDocument

            designspace = DesignSpaceDocument.fromfile(args.inputs[0])
            args.inputs = [master.path for master in designspace.sources]
            locations = [master.location for master in designspace.sources]
            axis_triples = {
                a.name: (a.minimum, a.default, a.maximum) for a in designspace.axes
            }
            axis_mappings = {a.name: a.map for a in designspace.axes}
            axis_triples = {
                k: tuple(piecewiseLinearMap(v, dict(axis_mappings[k])) for v in vv)
                for k, vv in axis_triples.items()
            }

        elif args.inputs[0].endswith(".glyphs"):
            from glyphsLib import GSFont, to_designspace

            gsfont = GSFont(args.inputs[0])
            designspace = to_designspace(gsfont)
            fonts = [source.font for source in designspace.sources]
            names = ["%s-%s" % (f.info.familyName, f.info.styleName) for f in fonts]
            args.inputs = []
            locations = [master.location for master in designspace.sources]
            axis_triples = {
                a.name: (a.minimum, a.default, a.maximum) for a in designspace.axes
            }
            axis_mappings = {a.name: a.map for a in designspace.axes}
            axis_triples = {
                k: tuple(piecewiseLinearMap(v, dict(axis_mappings[k])) for v in vv)
                for k, vv in axis_triples.items()
            }

        elif args.inputs[0].endswith(".ttf"):
            from fontTools.ttLib import TTFont

            font = TTFont(args.inputs[0])
            if "gvar" in font:
                # Is variable font

                axisMapping = {}
                fvar = font["fvar"]
                for axis in fvar.axes:
                    axisMapping[axis.axisTag] = {
                        -1: axis.minValue,
                        0: axis.defaultValue,
                        1: axis.maxValue,
                    }
                if "avar" in font:
                    avar = font["avar"]
                    for axisTag, segments in avar.segments.items():
                        fvarMapping = axisMapping[axisTag].copy()
                        for location, value in segments.items():
                            axisMapping[axisTag][value] = piecewiseLinearMap(
                                location, fvarMapping
                            )

                gvar = font["gvar"]
                glyf = font["glyf"]
                # Gather all glyphs at their "master" locations
                ttGlyphSets = {}
                glyphsets = defaultdict(dict)

                if glyphs is None:
                    glyphs = sorted(gvar.variations.keys())
                for glyphname in glyphs:
                    for var in gvar.variations[glyphname]:
                        locDict = {}
                        loc = []
                        for tag, val in sorted(var.axes.items()):
                            locDict[tag] = val[1]
                            loc.append((tag, val[1]))

                        locTuple = tuple(loc)
                        if locTuple not in ttGlyphSets:
                            ttGlyphSets[locTuple] = font.getGlyphSet(
                                location=locDict, normalized=True, recalcBounds=False
                            )

                        recursivelyAddGlyph(
                            glyphname, glyphsets[locTuple], ttGlyphSets[locTuple], glyf
                        )

                names = ["''"]
                fonts = [font.getGlyphSet()]
                locations = [{}]
                axis_triples = {a: (-1, 0, +1) for a in sorted(axisMapping.keys())}
                for locTuple in sorted(glyphsets.keys(), key=lambda v: (len(v), v)):
                    name = (
                        "'"
                        + " ".join(
                            "%s=%s"
                            % (
                                k,
                                floatToFixedToStr(
                                    piecewiseLinearMap(v, axisMapping[k]), 14
                                ),
                            )
                            for k, v in locTuple
                        )
                        + "'"
                    )
                    names.append(name)
                    fonts.append(glyphsets[locTuple])
                    locations.append(dict(locTuple))
                args.ignore_missing = True
                args.inputs = []

    if not locations:
        locations = [{} for _ in fonts]

    for filename in args.inputs:
        if filename.endswith(".ufo"):
            from fontTools.ufoLib import UFOReader

            fonts.append(UFOReader(filename))
        else:
            from fontTools.ttLib import TTFont

            fonts.append(TTFont(filename))

        names.append(basename(filename).rsplit(".", 1)[0])

    glyphsets = []
    for font in fonts:
        if hasattr(font, "getGlyphSet"):
            glyphset = font.getGlyphSet()
        else:
            glyphset = font
        glyphsets.append({k: glyphset[k] for k in glyphset.keys()})

    if args.name:
        accepted_names = set(args.name)
        glyphsets = [
            glyphset
            for name, glyphset in zip(names, glyphsets)
            if name in accepted_names
        ]
        locations = [
            location
            for name, location in zip(names, locations)
            if name in accepted_names
        ]
        names = [name for name in names if name in accepted_names]

    if not glyphs:
        glyphs = sorted(set([gn for glyphset in glyphsets for gn in glyphset.keys()]))

    glyphsSet = set(glyphs)
    for glyphset in glyphsets:
        glyphSetGlyphNames = set(glyphset.keys())
        diff = glyphsSet - glyphSetGlyphNames
        if diff:
            for gn in diff:
                glyphset[gn] = None

    # Normalize locations
    locations = [normalizeLocation(loc, axis_triples) for loc in locations]

    try:
        log.info("Running on %d glyphsets", len(glyphsets))
        log.info("Locations: %s", pformat(locations))
        problems_gen = test_gen(
            glyphsets,
            glyphs=glyphs,
            names=names,
            locations=locations,
            ignore_missing=args.ignore_missing,
            tolerance=args.tolerance or 0.95,
            show_all=args.show_all,
        )
        problems = defaultdict(list)

        f = sys.stdout if args.output is None else open(args.output, "w")

        if not args.quiet:
            if args.json:
                import json

                for glyphname, problem in problems_gen:
                    problems[glyphname].append(problem)

                print(json.dumps(problems), file=f)
            else:
                last_glyphname = None
                for glyphname, p in problems_gen:
                    problems[glyphname].append(p)

                    if glyphname != last_glyphname:
                        print(f"Glyph {glyphname} was not compatible:", file=f)
                        last_glyphname = glyphname
                        last_master_idxs = None

                    master_idxs = (
                        (p["master_idx"])
                        if "master_idx" in p
                        else (p["master_1_idx"], p["master_2_idx"])
                    )
                    if master_idxs != last_master_idxs:
                        master_names = (
                            (p["master"])
                            if "master" in p
                            else (p["master_1"], p["master_2"])
                        )
                        print(f"  Masters: %s:" % ", ".join(master_names), file=f)
                        last_master_idxs = master_idxs

                    if p["type"] == "missing":
                        print(
                            "    Glyph was missing in master %s" % p["master"], file=f
                        )
                    elif p["type"] == "open_path":
                        print(
                            "    Glyph has an open path in master %s" % p["master"],
                            file=f,
                        )
                    elif p["type"] == "path_count":
                        print(
                            "    Path count differs: %i in %s, %i in %s"
                            % (
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "node_count":
                        print(
                            "    Node count differs in path %i: %i in %s, %i in %s"
                            % (
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "node_incompatibility":
                        print(
                            "    Node %o incompatible in path %i: %s in %s, %s in %s"
                            % (
                                p["node"],
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "contour_order":
                        print(
                            "    Contour order differs: %s in %s, %s in %s"
                            % (
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "wrong_start_point":
                        print(
                            "    Contour %d start point differs: %s in %s, %s in %s; reversed: %s"
                            % (
                                p["contour"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                                p["reversed"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "wrong_structure":
                        print(
                            "    Contour %d structures differ: %s, %s"
                            % (
                                p["contour"],
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
                    elif p["type"] == "nothing":
                        print(
                            "    Nothing wrong between %s and %s"
                            % (
                                p["master_1"],
                                p["master_2"],
                            ),
                            file=f,
                        )
        else:
            for glyphname, problem in problems_gen:
                problems[glyphname].append(problem)

        if args.pdf:
            log.info("Writing PDF to %s", args.pdf)
            from .interpolatablePlot import InterpolatablePDF

            with InterpolatablePDF(args.pdf, glyphsets=glyphsets, names=names) as pdf:
                pdf.add_problems(problems)
                if not problems and not args.quiet:
                    pdf.draw_cupcake()

        if args.html:
            log.info("Writing HTML to %s", args.html)
            from .interpolatablePlot import InterpolatableSVG

            svgs = []
            with InterpolatableSVG(svgs, glyphsets=glyphsets, names=names) as svg:
                svg.add_problems(problems)
                if not problems and not args.quiet:
                    svg.draw_cupcake()

            import base64

            with open(args.html, "wb") as f:
                f.write(b"<!DOCTYPE html>\n")
                f.write(b"<html><body align=center>\n")
                for svg in svgs:
                    f.write("<img src='data:image/svg+xml;base64,".encode("utf-8"))
                    f.write(base64.b64encode(svg))
                    f.write(b"' />\n")
                    f.write(b"<hr>\n")
                f.write(b"</body></html>\n")

    except Exception as e:
        e.args += original_args_inputs
        log.error(e)
        raise

    if problems:
        return problems


if __name__ == "__main__":
    import sys

    problems = main()
    sys.exit(int(bool(problems)))
