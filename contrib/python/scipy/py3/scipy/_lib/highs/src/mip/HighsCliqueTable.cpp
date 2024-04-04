/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2022 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Leona Gottwald and Michael    */
/*    Feldmeier                                                          */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsCliqueTable.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>

#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"
#include "parallel/HighsCombinable.h"
#include "parallel/HighsParallel.h"
#include "pdqsort/pdqsort.h"
#include "presolve/HighsPostsolveStack.h"
#include "util/HighsSplay.h"

namespace highs {
template <>
struct RbTreeTraits<HighsCliqueTable::CliqueSet> {
  using KeyType = HighsInt;
  using LinkType = HighsInt;
};

}  // namespace highs

class HighsCliqueTable::CliqueSet : public highs::CacheMinRbTree<CliqueSet> {
  HighsCliqueTable* clqtable;

 public:
  CliqueSet(HighsCliqueTable* clqtable, CliqueVar v, bool sizeTwo = false)
      : highs::CacheMinRbTree<CliqueSet>(
            sizeTwo ? clqtable->sizeTwoCliquesetTree[v.index()].root
                    : clqtable->cliquesetTree[v.index()].root,
            sizeTwo ? clqtable->sizeTwoCliquesetTree[v.index()].first
                    : clqtable->cliquesetTree[v.index()].first),
        clqtable(clqtable) {}

  highs::RbTreeLinks<HighsInt>& getRbTreeLinks(HighsInt node) {
    return clqtable->cliquesets[node].links;
  }
  const highs::RbTreeLinks<HighsInt>& getRbTreeLinks(HighsInt node) const {
    return clqtable->cliquesets[node].links;
  }
  HighsInt getKey(HighsInt node) const {
    return clqtable->cliquesets[node].cliqueid;
  }
};

#define ADD_ZERO_WEIGHT_VARS

static std::pair<HighsCliqueTable::CliqueVar, HighsCliqueTable::CliqueVar>
sortedEdge(HighsCliqueTable::CliqueVar v1, HighsCliqueTable::CliqueVar v2) {
  if (v1.col > v2.col) return std::make_pair(v2, v1);

  return std::make_pair(v1, v2);
}

void HighsCliqueTable::unlink(HighsInt node) {
  assert(node >= 0);
  --numcliquesvar[cliqueentries[node].index()];

  HighsInt cliquelen = cliques[cliquesets[node].cliqueid].end -
                       cliques[cliquesets[node].cliqueid].start;
  CliqueSet clqset(this, cliqueentries[node], cliquelen == 2);

  clqset.unlink(node);
  cliquesets[node].cliqueid = -1;
}

void HighsCliqueTable::link(HighsInt node) {
  assert(node >= 0);
  ++numcliquesvar[cliqueentries[node].index()];
  assert(!colDeleted[cliqueentries[node].col]);

  HighsInt cliquelen = cliques[cliquesets[node].cliqueid].end -
                       cliques[cliquesets[node].cliqueid].start;
  CliqueSet clqset(this, cliqueentries[node], cliquelen == 2);

  clqset.link(node);
}

HighsInt HighsCliqueTable::findCommonCliqueId(int64_t& numQueries, CliqueVar v1,
                                              CliqueVar v2) {
  if (sizeTwoCliquesetTree[v1.index()].root != -1 &&
      sizeTwoCliquesetTree[v2.index()].root != -1) {
    ++numQueries;
    HighsInt* sizeTwoCliqueId = sizeTwoCliques.find(sortedEdge(v1, v2));
    if (sizeTwoCliqueId != nullptr) return *sizeTwoCliqueId;
  }

  CliqueSet v1Cliques(this, v1);
  CliqueSet v2Cliques(this, v2);

  if (v1Cliques.empty() || v2Cliques.empty()) return -1;

  ++numQueries;
  HighsInt i1 = v1Cliques.first();
  HighsInt v2LastClique = cliquesets[v2Cliques.last()].cliqueid;
  if (cliquesets[i1].cliqueid >= v2LastClique)
    return cliquesets[i1].cliqueid == v2LastClique ? v2LastClique : -1;

  HighsInt i2 = v2Cliques.first();
  HighsInt v1LastClique = cliquesets[v1Cliques.last()].cliqueid;
  if (cliquesets[i2].cliqueid >= v1LastClique)
    return cliquesets[i2].cliqueid == v1LastClique ? v1LastClique : -1;

  while (true) {
    if (cliquesets[i1].cliqueid < cliquesets[i2].cliqueid) {
      i1 = v1Cliques.successor(i1);
      if (i1 == -1) return -1;
      if (cliquesets[i1].cliqueid >= v2LastClique)
        return cliquesets[i1].cliqueid == v2LastClique ? v2LastClique : -1;
    } else if (cliquesets[i1].cliqueid > cliquesets[i2].cliqueid) {
      i2 = v2Cliques.successor(i2);
      if (i2 == -1) return -1;
      if (cliquesets[i2].cliqueid >= v1LastClique)
        return cliquesets[i2].cliqueid == v1LastClique ? v1LastClique : -1;
    } else
      return cliquesets[i1].cliqueid;

    ++numQueries;
  }
}

void HighsCliqueTable::resolveSubstitution(CliqueVar& v) const {
  while (colsubstituted[v.col]) {
    Substitution subst = substitutions[colsubstituted[v.col] - 1];
    v = v.val == 1 ? subst.replace : subst.replace.complement();
  }
}

void HighsCliqueTable::resolveSubstitution(HighsInt& col, double& val,
                                           double& offset) const {
  while (colsubstituted[col]) {
    Substitution subst = substitutions[colsubstituted[col] - 1];
    if (subst.replace.val == 0) {
      offset += val;
      val = -val;
    }
    col = subst.replace.col;
  }
}

HighsInt HighsCliqueTable::runCliqueSubsumption(
    const HighsDomain& globaldom, std::vector<CliqueVar>& clique) {
  if (clique.size() == 2) return 0;
  HighsInt nremoved = 0;
  bool redundant = false;
  if (cliquehits.size() < cliques.size()) cliquehits.resize(cliques.size());

  clique.erase(
      std::remove_if(clique.begin(), clique.end(),
                     [&](CliqueVar clqvar) { return colDeleted[clqvar.col]; }),
      clique.end());

  for (CliqueVar v : clique) {
    CliqueSet vClqs(this, v);
    HighsInt node = vClqs.first();
    while (node != -1) {
      HighsInt cliqueid = cliquesets[node].cliqueid;
      if (cliquehits[cliqueid] == 0) cliquehitinds.push_back(cliqueid);

      ++cliquehits[cliqueid];
      node = vClqs.successor(node);
    }

    CliqueSet vClqsSizeTwo = CliqueSet(this, v, true);

    node = vClqsSizeTwo.first();
    while (node != -1) {
      HighsInt cliqueid = cliquesets[node].cliqueid;
      if (cliquehits[cliqueid] == 0) cliquehitinds.push_back(cliqueid);

      ++cliquehits[cliqueid];
      node = vClqsSizeTwo.successor(node);
    }
  }

  for (HighsInt cliqueid : cliquehitinds) {
    HighsInt hits = cliquehits[cliqueid];
    cliquehits[cliqueid] = 0;

    HighsInt len = cliques[cliqueid].end - cliques[cliqueid].start -
                   cliques[cliqueid].numZeroFixed;
    if (hits == (HighsInt)clique.size())
      redundant = true;
    else if (len == hits) {
      if (cliques[cliqueid].equality) {
        for (CliqueVar v : clique) {
          bool sizeTwo = cliques[cliqueid].end - cliques[cliqueid].start == 2;
          CliqueSet vClqs = CliqueSet(this, v, sizeTwo);
          if (!vClqs.find(cliqueid).second) infeasvertexstack.push_back(v);
        }
      } else {
        ++nremoved;
        cliques[cliqueid].origin = kHighsIInf;
        removeClique(cliqueid);
      }
    }
  }

  cliquehitinds.clear();

  if (redundant) clique.clear();

  if (!infeasvertexstack.empty()) {
    clique.erase(
        std::remove_if(clique.begin(), clique.end(),
                       [&](CliqueVar v) { return globaldom.isFixed(v.col); }),
        clique.end());
  }

  return nremoved;
}

void HighsCliqueTable::bronKerboschRecurse(BronKerboschData& data,
                                           HighsInt Plen, const CliqueVar* X,
                                           HighsInt Xlen) {
  double w = data.wR;

  for (HighsInt i = 0; i != Plen; ++i) w += data.P[i].weight(data.sol);

  if (w < data.minW - data.feastol) return;

  if (Plen == 0 && Xlen == 0) {
    std::vector<CliqueVar> clique = data.R;

    if (data.minW < w - data.feastol) {
      data.maxcliques -= data.cliques.size();
      data.cliques.clear();
      data.minW = w;
    }
    data.cliques.emplace_back(std::move(clique));
    // do not further search for cliques that are violated less than this
    // current clique
    return;
  }

  ++data.ncalls;

  if (data.stop(numNeighborhoodQueries)) return;

  double pivweight = -1.0;
  CliqueVar pivot;

  for (HighsInt i = 0; i != Xlen; ++i) {
    if (X[i].weight(data.sol) > pivweight) {
      pivweight = X[i].weight(data.sol);
      pivot = X[i];
      if (pivweight >= 1.0 - data.feastol) break;
    }
  }

  if (pivweight < 1.0 - data.feastol) {
    for (HighsInt i = 0; i != Plen; ++i) {
      if (data.P[i].weight(data.sol) > pivweight) {
        pivweight = data.P[i].weight(data.sol);
        pivot = data.P[i];
        if (pivweight >= 1.0 - data.feastol) break;
      }
    }
  }

  std::vector<CliqueVar> PminusNu;
  PminusNu.reserve(Plen);
  queryNeighborhood(pivot, data.P.data(), Plen);
  neighborhoodInds.push_back(Plen);
  HighsInt k = 0;
  for (HighsInt i : neighborhoodInds) {
    while (k < i) PminusNu.push_back(data.P[k++]);
    ++k;
  }

  pdqsort(PminusNu.begin(), PminusNu.end(), [&](CliqueVar a, CliqueVar b) {
    return std::make_pair(a.weight(data.sol), a.index()) >
           std::make_pair(b.weight(data.sol), b.index());
  });

  std::vector<CliqueVar> localX;
  localX.insert(localX.end(), X, X + Xlen);

  for (CliqueVar v : PminusNu) {
    HighsInt newPlen = partitionNeighborhood(v, data.P.data(), Plen);
    HighsInt newXlen = partitionNeighborhood(v, localX.data(), localX.size());

    // add v to R, update the weight, and do the recursive call
    data.R.push_back(v);
    double wv = v.weight(data.sol);
    data.wR += wv;
    bronKerboschRecurse(data, newPlen, localX.data(), newXlen);
    if (data.stop(numNeighborhoodQueries)) return;

    // remove v from R restore the weight and continue the loop in this call
    data.R.pop_back();
    data.wR -= wv;

    w -= wv;
    if (w < data.minW) return;
    // find the position of v in the vertices removed from P for the recursive
    // call
    // and also remove it from the set P for this call
    HighsInt vpos = -1;
    for (HighsInt i = newPlen; i != Plen; ++i) {
      if (data.P[i] == v) {
        vpos = i;
        break;
      }
    }

    // do the removal by first swapping it to the end of P and reduce the size
    // of P accordingly
    assert(vpos != -1);

    --Plen;
    std::swap(data.P[vpos], data.P[Plen]);

    localX.push_back(v);
  }
}

#if 0
static void printRow(const HighsDomain& domain, const HighsInt* inds,
                     const double* vals, HighsInt len, double lhs, double rhs) {
  printf("%g <= ", lhs);

  for (HighsInt i = 0; i != len; ++i) {
    char sign = vals[i] > 0 ? '+' : '-';
    char var = domain.isBinary(inds[i]) ? 'x' : 'y';
    printf("%c%g %c%" HIGHSINT_FORMAT " ", sign, std::abs(vals[i]), var, inds[i]);
  }

  printf("<= %g\n", rhs);
}

static void printClique(
    const std::vector<HighsCliqueTable::CliqueVar>& clique) {
  bool first = true;
  for (HighsCliqueTable::CliqueVar v : clique) {
    if (!first) printf("+ ");
    char complemented = v.val == 0 ? '~' : ' ';
    printf("%cx%" HIGHSINT_FORMAT " ", complemented, v.col);
    first = false;
  }

  printf("<= 1\n");
}
#endif

void HighsCliqueTable::doAddClique(const CliqueVar* cliquevars,
                                   HighsInt numcliquevars, bool equality,
                                   HighsInt origin) {
  HighsInt cliqueid;

  if (freeslots.empty()) {
    cliqueid = cliques.size();
    cliques.emplace_back();
  } else {
    cliqueid = freeslots.back();
    freeslots.pop_back();
  }

  cliques[cliqueid].equality = equality;
  cliques[cliqueid].origin = origin;

  std::set<std::pair<HighsInt, int>>::iterator it;
  HighsInt maxEnd;
  if (freespaces.empty() || (it = freespaces.lower_bound(std::make_pair(
                                 numcliquevars, -1))) == freespaces.end()) {
    cliques[cliqueid].start = cliqueentries.size();
    cliques[cliqueid].end = cliques[cliqueid].start + numcliquevars;
    maxEnd = cliques[cliqueid].end;
    cliqueentries.resize(cliques[cliqueid].end);
    cliquesets.resize(cliques[cliqueid].end);
  } else {
    std::pair<HighsInt, int> freespace = *it;
    freespaces.erase(it);

    cliques[cliqueid].start = freespace.second;
    cliques[cliqueid].end = cliques[cliqueid].start + numcliquevars;
    maxEnd = cliques[cliqueid].start + freespace.first;
  }

  cliques[cliqueid].numZeroFixed = 0;

  bool fixtozero = false;
  HighsInt k = cliques[cliqueid].start;
  for (HighsInt i = 0; i != numcliquevars; ++i) {
    CliqueVar v = cliquevars[i];

    resolveSubstitution(v);

    if (fixtozero) {
      infeasvertexstack.push_back(v);
      continue;
    }

    // due to substitutions the variable may occur together with its complement
    // in this clique and we can fix all other variables in the clique to zero:
    //          x + ~x + ... <= 1
    //   <=> x + 1 - x + ... <= 1
    //   <=>             ... <= 0
    CliqueSet vComplClqs = CliqueSet(this, v.complement(), numcliquevars == 2);
    if (vComplClqs.find(cliqueid).second) {
      fixtozero = true;
      for (HighsInt j = cliques[cliqueid].start; j != k; ++j) {
        if (cliqueentries[j].col != v.col)
          infeasvertexstack.push_back(cliqueentries[j]);
        unlink(j);
      }
      k = cliques[cliqueid].start;
      continue;
    }

    // due to substitutions the variable may occur twice in this clique and
    // we can fix it to zero:  x + x + ... <= 1  <=>  2x <= 1 <=> x <= 0.5 <=>
    // x = 0
    CliqueSet vClqs = CliqueSet(this, v, numcliquevars == 2);
    std::pair<HighsInt, bool> vFindResult = vClqs.find(cliqueid);
    if (vFindResult.second) {
      infeasvertexstack.push_back(v);
      continue;
    }

    cliqueentries[k] = v;
    cliquesets[k].cliqueid = cliqueid;
    vClqs.link(k, vFindResult.first);
    ++numcliquesvar[v.index()];
    ++k;
  }

  if (maxEnd > k) {
    if (int(cliqueentries.size()) == maxEnd) {
      cliqueentries.resize(k);
      cliquesets.resize(k);
    } else
      freespaces.emplace(maxEnd - k, k);

    if (cliques[cliqueid].end > k) {
      switch (k - cliques[cliqueid].start) {
        case 0:
          // clique empty, so just mark it as deleted
          cliques[cliqueid].start = -1;
          cliques[cliqueid].end = -1;
          freeslots.push_back(cliqueid);
          return;
        case 1:
          // size 1 clique is redundant, so unlink the single linked entry
          // and mark it as deleted
          unlink(cliques[cliqueid].start);
          cliques[cliqueid].start = -1;
          cliques[cliqueid].end = -1;
          freeslots.push_back(cliqueid);
          return;
        case 2:
          // due to subsitutions the clique became smaller and is now of size
          // two as a result we need to link it to the size two cliqueset
          // instead of the normal cliqueset
          assert(cliqueid >= 0 && cliqueid < (HighsInt)cliques.size());
          assert(cliques[cliqueid].start >= 0 &&
                 cliques[cliqueid].start < (HighsInt)cliqueentries.size());
          unlink(cliques[cliqueid].start);
          unlink(cliques[cliqueid].start + 1);

          cliques[cliqueid].end = k;

          cliquesets[cliques[cliqueid].start].cliqueid = cliqueid;
          link(cliques[cliqueid].start);
          cliquesets[cliques[cliqueid].start + 1].cliqueid = cliqueid;
          link(cliques[cliqueid].start + 1);
          break;
        default:
          cliques[cliqueid].end = k;
      }
    }
  }

  HighsInt cliqueLen = cliques[cliqueid].end - cliques[cliqueid].start;
  numEntries += cliqueLen;
  if (cliqueLen == 2)
    sizeTwoCliques.insert(
        sortedEdge(cliqueentries[cliques[cliqueid].start],
                   cliqueentries[cliques[cliqueid].start + 1]),
        cliqueid);
}
struct ThreadNeighborhoodQueryData {
  int64_t numQueries;
  std::vector<HighsInt> neighborhoodInds;
};

void HighsCliqueTable::queryNeighborhood(CliqueVar v, CliqueVar* q,
                                         HighsInt N) {
  neighborhoodInds.clear();
  if (sizeTwoCliquesetTree[v.index()].root == -1 &&
      cliquesetTree[v.index()].root == -1)
    return;

  if (numEntries - sizeTwoCliques.size() * 2 < minEntriesForParallelism) {
    for (HighsInt i = 0; i < N; ++i) {
      if (haveCommonClique(numNeighborhoodQueries, v, q[i]))
        neighborhoodInds.push_back(i);
    }
  } else {
    auto neighborhoodData =
        makeHighsCombinable<ThreadNeighborhoodQueryData>([N]() {
          ThreadNeighborhoodQueryData d;
          d.neighborhoodInds.reserve(N);
          d.numQueries = 0;
          return d;
        });
    highs::parallel::for_each(
        0, N,
        [this, &neighborhoodData, v, q](HighsInt start, HighsInt end) {
          ThreadNeighborhoodQueryData& d = neighborhoodData.local();
          for (HighsInt i = start; i < end; ++i) {
            if (haveCommonClique(d.numQueries, v, q[i]))
              d.neighborhoodInds.push_back(i);
          }
        },
        10);

    neighborhoodData.combine_each([&](ThreadNeighborhoodQueryData& d) {
      neighborhoodInds.insert(neighborhoodInds.end(),
                              d.neighborhoodInds.begin(),
                              d.neighborhoodInds.end());
      numNeighborhoodQueries += d.numQueries;
    });
    pdqsort(neighborhoodInds.begin(), neighborhoodInds.end());
  }
}

HighsInt HighsCliqueTable::partitionNeighborhood(CliqueVar v, CliqueVar* q,
                                                 HighsInt N) {
  queryNeighborhood(v, q, N);

  for (HighsInt i = 0; i < (HighsInt)neighborhoodInds.size(); ++i)
    std::swap(q[i], q[neighborhoodInds[i]]);

  return neighborhoodInds.size();
}

HighsInt HighsCliqueTable::shrinkToNeighborhood(CliqueVar v, CliqueVar* q,
                                                HighsInt N) {
  queryNeighborhood(v, q, N);

  for (HighsInt i = 0; i < (HighsInt)neighborhoodInds.size(); ++i)
    q[i] = q[neighborhoodInds[i]];

  return neighborhoodInds.size();
}

bool HighsCliqueTable::processNewEdge(HighsDomain& globaldom, CliqueVar v1,
                                      CliqueVar v2) {
  if (v1.col == v2.col) {
    if (v1.val == v2.val) {
      bool wasfixed = globaldom.isFixed(v1.col);
      globaldom.fixCol(v1.col, double(1 - v1.val));
      if (!wasfixed) {
        ++nfixings;
        infeasvertexstack.push_back(v1);
        processInfeasibleVertices(globaldom);
      }
      return false;
    }

    return true;
  }

  // invertedEdgeCache.erase(sortedEdge(v1, v2));

  if (haveCommonClique(v1.complement(), v2)) {
    bool wasfixed = globaldom.isFixed(v2.col);
    globaldom.fixCol(v2.col, double(1 - v2.val));
    if (!wasfixed) {
      ++nfixings;
      infeasvertexstack.push_back(v2);
      processInfeasibleVertices(globaldom);
    }
    return false;
  } else if (haveCommonClique(v2.complement(), v1)) {
    bool wasfixed = globaldom.isFixed(v1.col);
    globaldom.fixCol(v1.col, double(1 - v1.val));
    if (!wasfixed) {
      ++nfixings;
      infeasvertexstack.push_back(v1);
      processInfeasibleVertices(globaldom);
    }
    return false;
  } else {
    HighsInt commonclique =
        findCommonCliqueId(v1.complement(), v2.complement());
    if (commonclique == -1) return false;

    while (commonclique != -1) {
      HighsInt start = cliques[commonclique].start;
      HighsInt end = cliques[commonclique].end;

      for (HighsInt i = start; i != end; ++i) {
        if (cliqueentries[i] == v1.complement() ||
            cliqueentries[i] == v2.complement())
          continue;

        bool wasfixed = globaldom.isFixed(cliqueentries[i].col);
        globaldom.fixCol(cliqueentries[i].col, 1 - cliqueentries[i].val);
        if (globaldom.infeasible()) return true;
        if (!wasfixed) {
          ++nfixings;
          infeasvertexstack.emplace_back(cliqueentries[i]);
        }
      }

      removeClique(commonclique);
      commonclique = findCommonCliqueId(v1.complement(), v2.complement());
    }

    processInfeasibleVertices(globaldom);
    if (globaldom.infeasible()) return false;

    commonclique = findCommonCliqueId(v1, v2);

    while (commonclique != -1) {
      HighsInt start = cliques[commonclique].start;
      HighsInt end = cliques[commonclique].end;

      for (HighsInt i = start; i != end; ++i) {
        if (cliqueentries[i] == v1 || cliqueentries[i] == v2) continue;

        bool wasfixed = globaldom.isFixed(cliqueentries[i].col);
        globaldom.fixCol(cliqueentries[i].col, 1 - cliqueentries[i].val);
        if (globaldom.infeasible()) return true;
        if (!wasfixed) {
          ++nfixings;
          infeasvertexstack.emplace_back(cliqueentries[i]);
        }
      }

      removeClique(commonclique);
      commonclique = findCommonCliqueId(v1, v2);
    }

    processInfeasibleVertices(globaldom);

    if (globaldom.isFixed(v1.col) || globaldom.isFixed(v2.col) ||
        globaldom.infeasible())
      return true;

    Substitution substitution;
    if (v2.col < v1.col) {
      if (v1.val == 1) v2 = v2.complement();

      substitution.substcol = v1.col;
      substitution.replace = v2;
    } else {
      if (v2.val == 1) v1 = v1.complement();

      substitution.substcol = v2.col;
      substitution.replace = v1;
    }

    substitutions.push_back(substitution);
    colsubstituted[substitution.substcol] = substitutions.size();

    HighsInt origindex = CliqueVar(substitution.substcol, 1).index();
    while (cliquesetTree[origindex].root != -1) {
      HighsInt node = cliquesetTree[origindex].root;
      HighsInt cliqueid = cliquesets[node].cliqueid;
      unlink(node);
      cliquesets[node].cliqueid = cliqueid;
      cliqueentries[node] = substitution.replace;
      link(node);
    }

    while (sizeTwoCliquesetTree[origindex].root != -1) {
      HighsInt node = sizeTwoCliquesetTree[origindex].root;
      HighsInt cliqueid = cliquesets[node].cliqueid;
      int othernode;
      if (node > 0 && cliquesets[node - 1].cliqueid == cliqueid)
        othernode = node - 1;
      else
        othernode = node + 1;

      assert(cliquesets[othernode].cliqueid == cliqueid && othernode != node);

      sizeTwoCliques.erase(
          sortedEdge(cliqueentries[node], cliqueentries[othernode]));
      unlink(node);
      cliquesets[node].cliqueid = cliqueid;
      cliqueentries[node] = substitution.replace;
      link(node);
      sizeTwoCliques.insert(
          sortedEdge(cliqueentries[node], cliqueentries[othernode]), cliqueid);
    }

    HighsInt complindex = CliqueVar(substitution.substcol, 0).index();
    while (cliquesetTree[complindex].root != -1) {
      HighsInt node = cliquesetTree[complindex].root;
      HighsInt cliqueid = cliquesets[node].cliqueid;
      unlink(node);
      cliquesets[node].cliqueid = cliqueid;
      cliqueentries[node] = substitution.replace.complement();
      link(node);
    }

    while (sizeTwoCliquesetTree[complindex].root != -1) {
      HighsInt node = sizeTwoCliquesetTree[complindex].root;
      HighsInt cliqueid = cliquesets[node].cliqueid;
      int othernode;
      if (node > 0 && cliquesets[node - 1].cliqueid == cliqueid)
        othernode = node - 1;
      else
        othernode = node + 1;

      assert(cliquesets[othernode].cliqueid == cliqueid && othernode != node);

      sizeTwoCliques.erase(
          sortedEdge(cliqueentries[node], cliqueentries[othernode]));
      unlink(node);
      cliquesets[node].cliqueid = cliqueid;
      cliqueentries[node] = substitution.replace.complement();
      link(node);
      sizeTwoCliques.insert(
          sortedEdge(cliqueentries[node], cliqueentries[othernode]), cliqueid);
    }

    return true;
  }
}

void HighsCliqueTable::addClique(const HighsMipSolver& mipsolver,
                                 CliqueVar* cliquevars, HighsInt numcliquevars,
                                 bool equality, HighsInt origin) {
  HighsDomain& globaldom = mipsolver.mipdata_->domain;
  mipsolver.mipdata_->debugSolution.checkClique(cliquevars, numcliquevars);
  for (HighsInt i = 0; i != numcliquevars; ++i) {
    resolveSubstitution(cliquevars[i]);
    if (globaldom.isFixed(cliquevars[i].col)) {
      if (cliquevars[i].val == globaldom.col_lower_[cliquevars[i].col]) {
        // column is fixed to 1, every other entry can be fixed to zero
        HighsInt k;
        for (k = 0; k < i; ++k) {
          bool wasfixed = globaldom.isFixed(cliquevars[k].col);
          globaldom.fixCol(cliquevars[k].col, double(1 - cliquevars[k].val));
          if (globaldom.infeasible()) return;
          if (!wasfixed) {
            ++nfixings;
            infeasvertexstack.push_back(cliquevars[k]);
          }
        }
        for (k = i + 1; k < numcliquevars; ++k) {
          bool wasfixed = globaldom.isFixed(cliquevars[k].col);
          globaldom.fixCol(cliquevars[k].col, double(1 - cliquevars[k].val));
          if (globaldom.infeasible()) return;
          if (!wasfixed) {
            ++nfixings;
            infeasvertexstack.push_back(cliquevars[k]);
          }
        }
      }

      processInfeasibleVertices(globaldom);
      return;
    }
  }

  if (numcliquevars <= 100) {
    bool hasNewEdge = false;

    // todo, sort new clique to allow log n lookup of membership in size by
    // binary search

    for (HighsInt i = 0; i < numcliquevars - 1; ++i) {
      if (globaldom.isFixed(cliquevars[i].col)) continue;
      if (cliquesetTree[cliquevars[i].index()].root == -1 &&
          sizeTwoCliquesetTree[cliquevars[i].index()].root == -1 &&
          cliquesetTree[cliquevars[i].complement().index()].root == -1 &&
          sizeTwoCliquesetTree[cliquevars[i].complement().index()].root == -1) {
        hasNewEdge = true;
        continue;
      }

      for (HighsInt j = i + 1; j < numcliquevars; ++j) {
        if (globaldom.isFixed(cliquevars[j].col)) continue;

        if (haveCommonClique(cliquevars[i], cliquevars[j])) continue;
        // todo: Instead of haveCommonClique use findCommonClique. If the common
        // clique is smaller than this clique check if it is a subset of this
        // clique. If it is a subset remove the clique and iterate the process
        // until either a common clique that is not a subset of this one is
        // found, or no common clique exists anymore in which case we proceed
        // with the code below and set hasNewEdge to true

        hasNewEdge = true;

        bool iscover = processNewEdge(globaldom, cliquevars[i], cliquevars[j]);
        if (globaldom.infeasible()) return;

        if (!mipsolver.mipdata_->nodequeue.empty()) {
          const auto& v1Nodes =
              cliquevars[i].val == 1
                  ? mipsolver.mipdata_->nodequeue.getUpNodes(cliquevars[i].col)
                  : mipsolver.mipdata_->nodequeue.getDownNodes(
                        cliquevars[i].col);
          const auto& v2Nodes =
              cliquevars[j].val == 1
                  ? mipsolver.mipdata_->nodequeue.getUpNodes(cliquevars[j].col)
                  : mipsolver.mipdata_->nodequeue.getDownNodes(
                        cliquevars[j].col);

          if (!v1Nodes.empty() && !v2Nodes.empty()) {
            // general integer variables can become binary during the search
            // and the cliques might be discovered. That means we need to take
            // care here, since the set of nodes branched upwards or downwards
            // are not necessarily containing domain changes setting the
            // variables to the corresponding clique value but could be
            // redundant bound changes setting the upper bound to u >= 1 or the
            // lower bound to l <= 0.

            // itV1 will point to the first node where v1 is fixed to val and
            // endV1 to the end of the range of such nodes. Same for itV2/endV2
            // with v2.
            auto itV1 = v1Nodes.lower_bound(
                std::make_pair(double(cliquevars[i].val), kHighsIInf));
            auto endV1 = v1Nodes.upper_bound(
                std::make_pair(double(cliquevars[i].val), kHighsIInf));
            auto itV2 = v2Nodes.lower_bound(
                std::make_pair(double(cliquevars[j].val), kHighsIInf));
            auto endV2 = v2Nodes.upper_bound(
                std::make_pair(double(cliquevars[j].val), kHighsIInf));

            if (itV1 != endV1 && itV2 != endV2 &&
                (itV1->second <= std::prev(endV2)->second ||
                 itV2->second <= std::prev(endV1)->second)) {
              // node ranges overlap, check for nodes that can be pruned
              while (itV1 != endV1 && itV2 != endV2) {
                if (itV1->second < itV2->second) {
                  ++itV1;
                } else if (itV2->second < itV1->second) {
                  ++itV2;
                } else {
                  // if (!mipsolver.submip)
                  //   printf("node %d can be pruned\n", itV2->second);
                  HighsInt prunedNode = itV2->second;
                  ++itV1;
                  ++itV2;
                  mipsolver.mipdata_->pruned_treeweight +=
                      mipsolver.mipdata_->nodequeue.pruneNode(prunedNode);
                }
              }
            }
          }
        }

        if (iscover) {
          for (HighsInt k = 0; k != numcliquevars; ++k) {
            if (k == i || k == j) continue;

            bool wasfixed = globaldom.isFixed(cliquevars[k].col);
            globaldom.fixCol(cliquevars[k].col, double(1 - cliquevars[k].val));
            if (globaldom.infeasible()) return;
            if (!wasfixed) {
              ++nfixings;
              infeasvertexstack.push_back(cliquevars[k]);
            }
          }

          processInfeasibleVertices(globaldom);
          return;
        }
      }
    }
    if (!hasNewEdge && origin == kHighsIInf) return;
  }
  CliqueVar* unfixedend =
      std::remove_if(cliquevars, cliquevars + numcliquevars,
                     [&](CliqueVar v) { return globaldom.isFixed(v.col); });
  numcliquevars = unfixedend - cliquevars;
  if (numcliquevars < 2) return;

  doAddClique(cliquevars, numcliquevars, equality, origin);
  processInfeasibleVertices(globaldom);
}

void HighsCliqueTable::removeClique(HighsInt cliqueid) {
  if (cliques[cliqueid].origin != kHighsIInf && cliques[cliqueid].origin != -1)
    deletedrows.push_back(cliques[cliqueid].origin);

  HighsInt start = cliques[cliqueid].start;
  assert(start != -1);
  HighsInt end = cliques[cliqueid].end;
  HighsInt len = end - start;
  if (len == 2) {
    sizeTwoCliques.erase(
        sortedEdge(cliqueentries[start], cliqueentries[start + 1]));
  }

  for (HighsInt i = start; i != end; ++i) {
    unlink(i);
  }

  freeslots.push_back(cliqueid);
  freespaces.emplace(len, start);

  cliques[cliqueid].start = -1;
  cliques[cliqueid].end = -1;
  numEntries -= len;
}

void HighsCliqueTable::extractCliques(
    const HighsMipSolver& mipsolver, std::vector<HighsInt>& inds,
    std::vector<double>& vals, std::vector<int8_t>& complementation, double rhs,
    HighsInt nbin, std::vector<HighsInt>& perm, std::vector<CliqueVar>& clique,
    double feastol) {
  HighsImplications& implics = mipsolver.mipdata_->implications;
  HighsDomain& globaldom = mipsolver.mipdata_->domain;

  perm.resize(inds.size());
  std::iota(perm.begin(), perm.end(), 0);

  auto binaryend = std::partition(perm.begin(), perm.end(), [&](HighsInt pos) {
    return globaldom.isBinary(inds[pos]);
  });
  nbin = binaryend - perm.begin();
  HighsInt ntotal = (HighsInt)perm.size();

  // if not all variables are binary, we extract variable upper and lower bounds
  // constraints on the non-binary variable for each binary variable in the
  // constraint
  if (nbin < ntotal) {
    for (HighsInt i = 0; i != nbin; ++i) {
      HighsInt bincol = inds[perm[i]];
      HighsCDouble impliedub = HighsCDouble(rhs) - vals[perm[i]];
      for (HighsInt j = nbin; j != ntotal; ++j) {
        HighsInt col = inds[perm[j]];
        if (globaldom.isFixed(col)) continue;

        HighsCDouble colub =
            HighsCDouble(globaldom.col_upper_[col]) - globaldom.col_lower_[col];
        HighsCDouble implcolub = impliedub / vals[perm[j]];
        if (implcolub < colub - feastol) {
          HighsCDouble coef;
          HighsCDouble constant;

          if (complementation[perm[i]] == -1) {
            coef = colub - implcolub;
            constant = implcolub;
          } else {
            coef = implcolub - colub;
            constant = colub;
          }

          if (complementation[perm[j]] == -1) {
            constant -= globaldom.col_upper_[col];
            implics.addVLB(col, bincol, -double(coef), -double(constant));
          } else {
            constant += globaldom.col_lower_[col];
            implics.addVUB(col, bincol, double(coef), double(constant));
          }
        }
      }
    }
  }

  // only one binary means we do have no cliques
  if (nbin <= 1) return;

  pdqsort(perm.begin(), binaryend, [&](HighsInt p1, HighsInt p2) {
    return std::make_pair(vals[p1], p1) > std::make_pair(vals[p2], p2);
  });
  // check if any cliques exists
  if (vals[perm[0]] + vals[perm[1]] <= rhs + feastol) return;

  // check if this is a set packing constraint (or easily transformable
  // into one)
  if (std::abs(vals[0] - vals[perm[nbin - 1]]) <= feastol &&
      rhs < 2 * vals[perm[nbin - 1]] - feastol) {
    // the coefficients on the binary variables are all equal and the
    // right hand side is strictly below two times the coefficient value.
    // Therefore the constraint can be transformed into a set packing
    // constraint by relaxing out all non-binary variables (if any),
    // dividing by the coefficient value of the binary variables, and then
    // possibly rounding down the right hand side
    clique.clear();

    for (auto j = 0; j != nbin; ++j) {
      HighsInt pos = perm[j];
      if (complementation[pos] == -1)
        clique.emplace_back(inds[pos], 0);
      else
        clique.emplace_back(inds[pos], 1);
    }

    addClique(mipsolver, clique.data(), nbin);
    if (globaldom.infeasible()) return;
    // printf("extracted this clique:\n");
    // printClique(clique);
    return;
  }

  for (HighsInt k = nbin - 1; k != 0; --k) {
    double mincliqueval = rhs - vals[perm[k]] + feastol;
    auto cliqueend = std::partition_point(
        perm.begin(), perm.begin() + k,
        [&](HighsInt p) { return vals[p] > mincliqueval; });

    // no clique for this variable
    if (cliqueend == perm.begin()) continue;

    clique.clear();

    for (auto j = perm.begin(); j != cliqueend; ++j) {
      HighsInt pos = *j;
      if (complementation[pos] == -1)
        clique.emplace_back(inds[pos], 0);
      else
        clique.emplace_back(inds[pos], 1);
    }

    if (complementation[perm[k]] == -1)
      clique.emplace_back(inds[perm[k]], 0);
    else
      clique.emplace_back(inds[perm[k]], 1);

    // printf("extracted this clique:\n");
    // printClique(clique);

    if (clique.size() >= 2) {
      // if (clique.size() > 2) runCliqueSubsumption(globaldom, clique);
      // runCliqueMerging(globaldom, clique);
      // if (clique.size() >= 2) {
      addClique(mipsolver, clique.data(), clique.size());
      if (globaldom.infeasible()) return;
      //}
    }

    // further cliques are just subsets of this clique
    if (cliqueend == perm.begin() + k) return;
  }
}

void HighsCliqueTable::cliquePartition(std::vector<CliqueVar>& clqVars,
                                       std::vector<HighsInt>& partitionStart) {
  randgen.shuffle(clqVars.data(), clqVars.size());

  HighsInt numClqVars = clqVars.size();
  partitionStart.clear();
  partitionStart.reserve(clqVars.size());
  HighsInt extensionEnd = numClqVars;
  partitionStart.push_back(0);
  for (HighsInt i = 0; i < numClqVars; ++i) {
    if (i == extensionEnd) {
      partitionStart.push_back(i);
      extensionEnd = numClqVars;
    }
    CliqueVar v = clqVars[i];
    HighsInt extensionStart = i + 1;
    extensionEnd = partitionNeighborhood(v, clqVars.data() + extensionStart,
                                         extensionEnd - extensionStart) +
                   extensionStart;
  }

  partitionStart.push_back(numClqVars);
}

void HighsCliqueTable::cliquePartition(const std::vector<double>& objective,
                                       std::vector<CliqueVar>& clqVars,
                                       std::vector<HighsInt>& partitionStart) {
  randgen.shuffle(clqVars.data(), clqVars.size());

  pdqsort_branchless(clqVars.begin(), clqVars.end(),
                     [&](CliqueVar v1, CliqueVar v2) {
                       return (2 * v1.val - 1) * objective[v1.col] >
                              (2 * v2.val - 1) * objective[v2.col];
                     });

  HighsInt numClqVars = clqVars.size();
  partitionStart.clear();
  partitionStart.reserve(clqVars.size());
  HighsInt extensionEnd = numClqVars;
  partitionStart.push_back(0);
  HighsInt lastSwappedIndex = 0;
  for (HighsInt i = 0; i < numClqVars; ++i) {
    if (i == extensionEnd) {
      partitionStart.push_back(i);
      extensionEnd = numClqVars;
      if (lastSwappedIndex >= i)
        pdqsort_branchless(clqVars.begin() + i,
                           clqVars.begin() + lastSwappedIndex + 1,
                           [&](CliqueVar v1, CliqueVar v2) {
                             return (2 * v1.val - 1) * objective[v1.col] >
                                    (2 * v2.val - 1) * objective[v2.col];
                           });
      lastSwappedIndex = 0;
    }
    CliqueVar v = clqVars[i];
    HighsInt extensionStart = i + 1;
    extensionEnd = partitionNeighborhood(v, clqVars.data() + extensionStart,
                                         extensionEnd - extensionStart) +
                   extensionStart;
    if (!neighborhoodInds.empty())
      lastSwappedIndex =
          std::max(neighborhoodInds.back() + extensionStart, lastSwappedIndex);
  }

  partitionStart.push_back(numClqVars);
}

bool HighsCliqueTable::foundCover(HighsDomain& globaldom, CliqueVar v1,
                                  CliqueVar v2) {
  bool equality = false;
  HighsInt commonclique = findCommonCliqueId(v1, v2);
  if (commonclique != -1) equality = true;

  while (commonclique != -1) {
    HighsInt start = cliques[commonclique].start;
    HighsInt end = cliques[commonclique].end;

    for (HighsInt i = start; i != end; ++i) {
      if (cliqueentries[i] == v1 || cliqueentries[i] == v2) continue;

      bool wasfixed = globaldom.isFixed(cliqueentries[i].col);
      globaldom.fixCol(cliqueentries[i].col, 1 - cliqueentries[i].val);
      if (globaldom.infeasible()) return equality;
      if (!wasfixed) {
        ++nfixings;
        infeasvertexstack.emplace_back(cliqueentries[i]);
      }
    }

    removeClique(commonclique);
    commonclique = findCommonCliqueId(v1, v2);
  }

  processInfeasibleVertices(globaldom);

  return equality;
}

void HighsCliqueTable::extractCliquesFromCut(const HighsMipSolver& mipsolver,
                                             const HighsInt* inds,
                                             const double* vals, HighsInt len,
                                             double rhs) {
  if (isFull()) return;

  HighsImplications& implics = mipsolver.mipdata_->implications;
  HighsDomain& globaldom = mipsolver.mipdata_->domain;

  const double feastol = mipsolver.mipdata_->feastol;

  HighsCDouble minact = 0.0;
  HighsInt nbin = 0;
  for (HighsInt i = 0; i != len; ++i) {
    if (globaldom.isBinary(inds[i])) ++nbin;

    if (vals[i] > 0) {
      if (globaldom.col_lower_[inds[i]] == -kHighsInf) return;
      minact += vals[i] * globaldom.col_lower_[inds[i]];
    } else {
      if (globaldom.col_upper_[inds[i]] == kHighsInf) return;
      minact += vals[i] * globaldom.col_upper_[inds[i]];
    }
  }

  for (HighsInt i = 0; i != len; ++i) {
    if (mipsolver.variableType(inds[i]) == HighsVarType::kContinuous) continue;

    double boundVal = double((rhs - minact) / vals[i]);
    if (vals[i] > 0) {
      boundVal = std::floor(boundVal + globaldom.col_lower_[inds[i]] +
                            globaldom.feastol());
      globaldom.changeBound(HighsBoundType::kUpper, inds[i], boundVal,
                            HighsDomain::Reason::unspecified());
      if (globaldom.infeasible()) return;
    } else {
      boundVal = std::ceil(boundVal + globaldom.col_upper_[inds[i]] -
                           globaldom.feastol());
      globaldom.changeBound(HighsBoundType::kLower, inds[i], boundVal,
                            HighsDomain::Reason::unspecified());
      if (globaldom.infeasible()) return;
    }
  }

  if (nbin <= 1) return;

  std::vector<HighsInt> perm;
  perm.resize(len);
  std::iota(perm.begin(), perm.end(), 0);

  auto binaryend = std::partition(perm.begin(), perm.end(), [&](HighsInt pos) {
    return globaldom.isBinary(inds[pos]);
  });

  nbin = binaryend - perm.begin();

  // if not all variables are binary, we extract variable upper and lower bounds
  // constraints on the non-binary variable for each binary variable in the
  // constraint:
  // todo@ evaluate performance impact, seems to cause slowdown on some
  // instances
  if (false && nbin < len) {
    for (HighsInt i = 0; i != nbin; ++i) {
      HighsInt bincol = inds[perm[i]];
      HighsCDouble impliedActivity = rhs - minact - std::abs(vals[perm[i]]);
      for (HighsInt j = nbin; j != len; ++j) {
        HighsInt col = inds[perm[j]];
        if (globaldom.isFixed(col)) continue;

        if (vals[perm[j]] > 0) {
          double implcolub = double(impliedActivity +
                                    vals[perm[j]] * globaldom.col_lower_[col]) /
                             vals[perm[j]];

          if (implcolub < globaldom.col_upper_[col] - feastol) {
            double coef;
            double constant;
            if (vals[perm[i]] < 0) {
              coef = globaldom.col_upper_[col] - implcolub;
              constant = implcolub;
            } else {
              coef = implcolub - globaldom.col_upper_[col];
              constant = globaldom.col_upper_[col];
            }
            // printf("extracted VUB from cut: x%" HIGHSINT_FORMAT " <= %g*y%"
            // HIGHSINT_FORMAT " + %g\n", col, coef,
            //        bincol, constant);
            implics.addVUB(col, bincol, coef, constant);
          }
        } else {
          double implcollb = double(impliedActivity +
                                    vals[perm[j]] * globaldom.col_upper_[col]) /
                             vals[perm[j]];
          if (implcollb > globaldom.col_lower_[col] + feastol) {
            double coef;
            double constant;
            if (vals[perm[i]] < 0) {
              coef = globaldom.col_lower_[col] - implcollb;
              constant = implcollb;
            } else {
              coef = implcollb - globaldom.col_lower_[col];
              constant = globaldom.col_lower_[col];
            }

            // printf("extracted VLB from cut: x%" HIGHSINT_FORMAT " >= %g*y%"
            // HIGHSINT_FORMAT " + %g\n", col, coef,
            //        bincol, constant);
            implics.addVLB(col, bincol, coef, constant);
            // printf("extracted VLB from cut\n");
          }
        }
      }
    }
  }

  // only one binary means we do have no cliques
  if (nbin <= 1) return;

  std::vector<CliqueVar> clique;
  clique.reserve(nbin);

  pdqsort(perm.begin(), binaryend, [&](HighsInt p1, HighsInt p2) {
    return std::make_pair(std::abs(vals[p1]), p1) >
           std::make_pair(std::abs(vals[p2]), p2);
  });
  // check if any cliques exists
  if (std::abs(vals[perm[0]]) + std::abs(vals[perm[1]]) <=
      double(rhs - minact + feastol))
    return;

  HighsInt maxNewEntries =
      std::min(mipsolver.mipdata_->numCliqueEntriesAfterPresolve + 100000 +
                   4 * globaldom.numModelNonzeros(),
               numEntries + 10 * nbin);

  for (HighsInt k = nbin - 1; k != 0 && numEntries < maxNewEntries; --k) {
    double mincliqueval =
        double(rhs - minact - std::abs(vals[perm[k]]) + feastol);
    auto cliqueend = std::partition_point(
        perm.begin(), perm.begin() + k,
        [&](HighsInt p) { return std::abs(vals[p]) > mincliqueval; });

    // no clique for this variable
    if (cliqueend == perm.begin()) continue;

    clique.clear();

    for (auto j = perm.begin(); j != cliqueend; ++j) {
      HighsInt pos = *j;
      if (vals[pos] < 0)
        clique.emplace_back(inds[pos], 0);
      else
        clique.emplace_back(inds[pos], 1);
    }

    if (vals[perm[k]] < 0)
      clique.emplace_back(inds[perm[k]], 0);
    else
      clique.emplace_back(inds[perm[k]], 1);

    // printf("extracted this clique:\n");
    // printClique(clique);
    if (clique.size() >= 2) {
      // printf("extracted clique from cut\n");
      // if (clique.size() > 2) runCliqueSubsumption(globaldom, clique);

      addClique(mipsolver, clique.data(), clique.size());
      if (globaldom.infeasible() || numEntries >= maxNewEntries) return;
    }

    // further cliques are just subsets of this clique
    if (cliqueend == perm.begin() + k) return;
  }
}

void HighsCliqueTable::extractCliques(HighsMipSolver& mipsolver,
                                      bool transformRows) {
  std::vector<HighsInt> inds;
  std::vector<double> vals;
  std::vector<HighsInt> perm;
  std::vector<int8_t> complementation;
  std::vector<CliqueVar> clique;
  HighsHashTable<HighsInt, double> entries;
  double offset;

  double rhs;

  HighsDomain& globaldom = mipsolver.mipdata_->domain;

  for (HighsInt i = 0; i != mipsolver.numRow(); ++i) {
    HighsInt start = mipsolver.mipdata_->ARstart_[i];
    HighsInt end = mipsolver.mipdata_->ARstart_[i + 1];

    if (mipsolver.mipdata_->postSolveStack.getOrigRowIndex(i) >=
        mipsolver.orig_model_->num_row_)
      break;

    // catch set packing and partitioning constraints that already have the form
    // of a clique without transformations and add those cliques with the rows
    // being recorded
    if (mipsolver.rowUpper(i) == 1.0) {
      bool issetppc = true;

      clique.clear();

      for (HighsInt j = start; j != end; ++j) {
        HighsInt col = mipsolver.mipdata_->ARindex_[j];
        if (globaldom.col_upper_[col] == 0.0 &&
            globaldom.col_lower_[col] == 0.0)
          continue;
        if (!globaldom.isBinary(col)) {
          issetppc = false;
          break;
        }

        if (mipsolver.mipdata_->ARvalue_[j] != 1.0) {
          issetppc = false;
          break;
        }

        clique.emplace_back(col, 1);
      }

      if (issetppc) {
        bool equality = mipsolver.rowLower(i) == 1.0;
        addClique(mipsolver, clique.data(), clique.size(), equality, i);
        if (globaldom.infeasible()) return;
        continue;
      }
    }
    if (!transformRows || isFull()) continue;

    offset = 0;
    for (HighsInt j = start; j != end; ++j) {
      HighsInt col = mipsolver.mipdata_->ARindex_[j];
      double val = mipsolver.mipdata_->ARvalue_[j];

      resolveSubstitution(col, val, offset);
      entries[col] += val;
    }

    if (mipsolver.rowUpper(i) != kHighsInf) {
      rhs = mipsolver.rowUpper(i) - offset;
      inds.clear();
      vals.clear();
      complementation.clear();
      bool freevar = false;
      HighsInt nbin = 0;

      for (const auto& entry : entries) {
        HighsInt col = entry.key();
        double val = entry.value();

        if (std::abs(val) < mipsolver.mipdata_->epsilon) continue;

        if (globaldom.isBinary(col)) ++nbin;

        if (val < 0) {
          if (globaldom.col_upper_[col] == kHighsInf) {
            freevar = true;
            break;
          }

          vals.push_back(-val);
          inds.push_back(col);
          complementation.push_back(-1);
          rhs -= val * globaldom.col_upper_[col];
        } else {
          if (globaldom.col_lower_[col] == -kHighsInf) {
            freevar = true;
            break;
          }

          vals.push_back(val);
          inds.push_back(col);
          complementation.push_back(1);
          rhs -= val * globaldom.col_lower_[col];
        }
      }

      if (!freevar && nbin != 0) {
        // printf("extracing cliques from this row:\n");
        // printRow(globaldom, inds.data(), vals.data(), inds.size(),
        //         -kHighsInf, rhs);
        extractCliques(mipsolver, inds, vals, complementation, rhs, nbin, perm,
                       clique, mipsolver.mipdata_->feastol);
        if (globaldom.infeasible()) return;
      }
    }

    if (mipsolver.rowLower(i) != -kHighsInf) {
      rhs = -mipsolver.rowLower(i) + offset;
      inds.clear();
      vals.clear();
      complementation.clear();
      bool freevar = false;
      HighsInt nbin = 0;

      for (const auto& entry : entries) {
        HighsInt col = entry.key();
        double val = -entry.value();
        if (std::abs(val) < mipsolver.mipdata_->epsilon) continue;

        if (globaldom.isBinary(col)) ++nbin;

        if (val < 0) {
          if (globaldom.col_upper_[col] == kHighsInf) {
            freevar = true;
            break;
          }

          vals.push_back(-val);
          inds.push_back(col);
          complementation.push_back(-1);
          rhs -= val * globaldom.col_upper_[col];
        } else {
          if (globaldom.col_lower_[col] == -kHighsInf) {
            freevar = true;
            break;
          }

          vals.push_back(val);
          inds.push_back(col);
          complementation.push_back(1);
          rhs -= val * globaldom.col_lower_[col];
        }
      }

      if (!freevar && nbin != 0) {
        // printf("extracing cliques from this row:\n");
        // printRow(globaldom, inds.data(), vals.data(), inds.size(),
        //         -kHighsInf, rhs);
        extractCliques(mipsolver, inds, vals, complementation, rhs, nbin, perm,
                       clique, mipsolver.mipdata_->feastol);
        if (globaldom.infeasible()) return;
      }
    }

    entries.clear();
  }
}

void HighsCliqueTable::extractObjCliques(HighsMipSolver& mipsolver) {
  HighsInt nbin =
      mipsolver.mipdata_->objectiveFunction.getNumBinariesInObjective();
  if (nbin <= 1) return;
  HighsDomain& globaldom = mipsolver.mipdata_->domain;
  if (globaldom.getObjectiveLowerBound() == -kHighsInf) return;

  const double* vals;
  const HighsInt* inds;
  HighsInt len;
  double rhs;
  globaldom.getCutoffConstraint(vals, inds, len, rhs);

  std::vector<HighsInt> perm;
  perm.resize(nbin);
  std::iota(perm.begin(), perm.end(), 0);

  auto binaryend = std::partition(perm.begin(), perm.end(), [&](HighsInt pos) {
    return vals[pos] != 0.0 && !globaldom.isFixed(inds[pos]);
  });

  nbin = binaryend - perm.begin();

  // only one binary means we do have no cliques
  if (nbin <= 1) return;

  std::vector<CliqueVar> clique;
  clique.reserve(nbin);

  pdqsort(perm.begin(), binaryend, [&](HighsInt p1, HighsInt p2) {
    return std::make_pair(std::fabs(vals[p1]), p1) >
           std::make_pair(std::fabs(vals[p2]), p2);
  });

  // check if any cliques exists
  HighsCDouble minact;
  HighsInt ninf;
  globaldom.computeMinActivity(0, len, inds, vals, ninf, minact);
  const double feastol = mipsolver.mipdata_->feastol;
  if (std::fabs(vals[perm[0]]) + std::fabs(vals[perm[1]]) <=
      double(rhs - minact + feastol))
    return;

  for (HighsInt k = nbin - 1; k != 0; --k) {
    double mincliqueval =
        double(rhs - minact - std::fabs(vals[perm[k]]) + feastol);
    auto cliqueend = std::partition_point(
        perm.begin(), perm.begin() + k,
        [&](HighsInt p) { return std::abs(vals[p]) > mincliqueval; });

    // no clique for this variable
    if (cliqueend == perm.begin()) continue;

    clique.clear();

    for (auto j = perm.begin(); j != cliqueend; ++j) {
      HighsInt pos = *j;
      if (vals[pos] < 0)
        clique.emplace_back(inds[pos], 0);
      else
        clique.emplace_back(inds[pos], 1);
    }

    if (vals[perm[k]] < 0)
      clique.emplace_back(inds[perm[k]], 0);
    else
      clique.emplace_back(inds[perm[k]], 1);

    // printf("extracted this clique from obj:\n");
    // printClique(clique);
    if (clique.size() >= 2) {
      // printf("extracted clique from obj\n");
      // if (clique.size() > 2) runCliqueSubsumption(globaldom, clique);

      addClique(mipsolver, clique.data(), clique.size());
      if (globaldom.infeasible()) return;
    }

    // further cliques are just subsets of this clique
    if (cliqueend == perm.begin() + k) return;
  }
}

void HighsCliqueTable::processInfeasibleVertices(HighsDomain& globaldom) {
  while (!infeasvertexstack.empty() && !globaldom.infeasible()) {
    CliqueVar v = infeasvertexstack.back().complement();
    infeasvertexstack.pop_back();

    resolveSubstitution(v);
    bool wasfixed = globaldom.isFixed(v.col);
    globaldom.fixCol(v.col, double(v.val));
    if (globaldom.infeasible()) return;
    if (!wasfixed) ++nfixings;
    if (colDeleted[v.col]) continue;
    colDeleted[v.col] = true;

    HighsInt node = cliquesetTree[v.index()].root != -1
                        ? cliquesetTree[v.index()].root
                        : sizeTwoCliquesetTree[v.index()].root;
    while (node != -1) {
      HighsInt cliqueid = cliquesets[node].cliqueid;
      HighsInt start = cliques[cliqueid].start;
      HighsInt end = cliques[cliqueid].end;

      for (HighsInt i = start; i != end; ++i) {
        if (cliqueentries[i].col == v.col) continue;

        bool wasfixed = globaldom.isFixed(cliqueentries[i].col);
        globaldom.fixCol(cliqueentries[i].col,
                         double(1 - cliqueentries[i].val));
        if (globaldom.infeasible()) return;
        if (!wasfixed) {
          ++nfixings;
          infeasvertexstack.push_back(cliqueentries[i]);
        }
      }

      removeClique(cliqueid);
      node = cliquesetTree[v.index()].root != -1
                 ? cliquesetTree[v.index()].root
                 : sizeTwoCliquesetTree[v.index()].root;
    }

    CliqueSet vComplClqs(this, v.complement());
    if (inPresolve) {
      // during presolve we only count the number of zeros within each clique
      // and only remove fully redundant cliques that are larger than two
      // in the process since during presolve a lot of cliques of size two
      // may be found by probing and will be deleted upon rebuild anyways
      node = vComplClqs.first();
      if (node == -1) continue;
      do {
        HighsInt cliqueid = cliquesets[node].cliqueid;
        assert(cliqueentries[node].val == 1 - v.val);
        node = vComplClqs.successor(node);

        cliques[cliqueid].numZeroFixed += 1;
        if (cliques[cliqueid].end - cliques[cliqueid].start -
                cliques[cliqueid].numZeroFixed <=
            1)
          removeClique(cliqueid);
      } while (node != -1);
      continue;
    }

    node = sizeTwoCliquesetTree[v.complement().index()].root;
    while (node != -1) {
      HighsInt cliqueid = cliquesets[node].cliqueid;
      removeClique(cliqueid);
      node = sizeTwoCliquesetTree[v.complement().index()].root;
    }

    assert(cliquehitinds.empty());
    node = vComplClqs.first();
    if (node == -1) continue;
    std::vector<CliqueVar> clq;
    do {
      HighsInt cliqueid = cliquesets[node].cliqueid;
      assert(cliqueentries[node].val == 1 - v.val);
      node = vComplClqs.successor(node);

      cliques[cliqueid].numZeroFixed += 1;
      if (cliques[cliqueid].end - cliques[cliqueid].start -
              cliques[cliqueid].numZeroFixed <=
          1) {
        removeClique(cliqueid);
      } else if (cliques[cliqueid].numZeroFixed >=
                 std::max(
                     HighsInt{10},
                     (cliques[cliqueid].end - cliques[cliqueid].start) >> 1)) {
        HighsInt initSize = cliques[cliqueid].end - cliques[cliqueid].start;
        clq.assign(cliqueentries.begin() + cliques[cliqueid].start,
                   cliqueentries.begin() + cliques[cliqueid].end);
        HighsInt numDel = 0;
        for (CliqueVar x : clq) numDel += colDeleted[x.col];

        assert(numDel == cliques[cliqueid].numZeroFixed);

        removeClique(cliqueid);
        clq.erase(std::remove_if(clq.begin(), clq.end(),
                                 [&](CliqueVar x) {
                                   return globaldom.isFixed(x.col) &&
                                          globaldom.col_lower_[x.col] ==
                                              1 - x.val;
                                 }),
                  clq.end());
        if (clq.size() > 1) doAddClique(clq.data(), clq.size());
      }
    } while (node != -1);
  }

  propagateAndCleanup(globaldom);
}

void HighsCliqueTable::propagateAndCleanup(HighsDomain& globaldom) {
  const auto& domchgstack = globaldom.getDomainChangeStack();
  HighsInt start = domchgstack.size();
  globaldom.propagate();
  HighsInt end = domchgstack.size();

  while (!globaldom.infeasible() && start != end) {
    for (HighsInt k = start; k != end; ++k) {
      HighsInt col = domchgstack[k].column;
      if (globaldom.col_lower_[col] != globaldom.col_upper_[col]) continue;
      if (globaldom.col_lower_[col] != 1.0 && globaldom.col_lower_[col] != 0.0)
        continue;

      HighsInt fixval = (HighsInt)globaldom.col_lower_[col];
      CliqueVar v(col, 1 - fixval);
      if (numcliquesvar[v.index()] != 0) {
        vertexInfeasible(globaldom, col, 1 - fixval);
        if (globaldom.infeasible()) return;
      }
    }
    start = domchgstack.size();
    globaldom.propagate();
    end = domchgstack.size();
  }
}

void HighsCliqueTable::vertexInfeasible(HighsDomain& globaldom, HighsInt col,
                                        HighsInt val) {
  bool wasfixed = globaldom.isFixed(col);
  globaldom.fixCol(col, double(1 - val));
  if (globaldom.infeasible()) return;
  if (!wasfixed) ++nfixings;
  infeasvertexstack.emplace_back(col, val);
  processInfeasibleVertices(globaldom);
}

void HighsCliqueTable::separateCliques(const HighsMipSolver& mipsolver,
                                       const std::vector<double>& sol,
                                       HighsCutPool& cutpool, double feastol) {
  BronKerboschData data(sol);
  data.feastol = feastol;
  data.maxNeighborhoodQueries = 10000000 +
                                int64_t{1000} * mipsolver.numNonzero() +
                                mipsolver.mipdata_->total_lp_iterations * 10000;
  if (numNeighborhoodQueries > data.maxNeighborhoodQueries) return;
  const HighsDomain& globaldom = mipsolver.mipdata_->domain;

  assert(numcliquesvar.size() == 2 * globaldom.col_lower_.size());
  for (HighsInt i : mipsolver.mipdata_->integral_cols) {
    if (colsubstituted[i] || colDeleted[i]) continue;
#ifdef ADD_ZERO_WEIGHT_VARS
    if (numcliquesvar[CliqueVar(i, 0).index()] != 0) {
      if (CliqueVar(i, 0).weight(sol) > feastol)
        data.P.emplace_back(i, 0);
      else
        data.Z.emplace_back(i, 0);
    }
    if (numcliquesvar[CliqueVar(i, 1).index()] != 0) {
      if (CliqueVar(i, 1).weight(sol) > feastol)
        data.P.emplace_back(i, 1);
      else
        data.Z.emplace_back(i, 1);
    }
#else
    if (numcliquesvar[CliqueVar(i, 0).index()] != 0 &&
        CliqueVar(i, 0).weight(sol) > feastol)
      data.P.emplace_back(i, 0);
    if (numcliquesvar[CliqueVar(i, 1).index()] != 0 &&
        CliqueVar(i, 1).weight(sol) > feastol)
      data.P.emplace_back(i, 1);
#endif
  }

  // auto t1 = std::chrono::high_resolution_clock::now();
  bronKerboschRecurse(data, data.P.size(), nullptr, 0);

  // auto t2 = std::chrono::high_resolution_clock::now();

  // printf(
  //     "bron kerbosch: %" HIGHSINT_FORMAT " calls, %" HIGHSINT_FORMAT "
  //     cliques, %ldms\n", data.ncalls, int(data.cliques.size()),
  //     std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //     t1).count());

  bool runcliquesubsumption = false;
  std::vector<HighsInt> inds;
  std::vector<double> vals;
  for (std::vector<CliqueVar>& clique : data.cliques) {
#ifdef ADD_ZERO_WEIGHT_VARS
    auto extensionend = data.Z.size();
    for (CliqueVar v : clique) {
      extensionend = partitionNeighborhood(v, data.Z.data(), extensionend);
      if (extensionend == 0) break;
    }

    if (extensionend != 0) {
      randgen.shuffle(data.Z.data(), extensionend);

      for (HighsInt i = 0; i < extensionend; ++i) {
        HighsInt k = i + 1;
        extensionend = k + partitionNeighborhood(data.Z[i], data.Z.data() + k,
                                                 extensionend - k);
      }

      clique.insert(clique.end(), data.Z.begin(),
                    data.Z.begin() + extensionend);
    }
#endif

    double rhs = 1;
    runcliquesubsumption = cliques.size() > 2;
    inds.clear();
    vals.clear();

    for (CliqueVar v : clique) {
      inds.push_back(v.col);
      if (v.val == 0) {
        vals.push_back(-1);
        rhs -= 1;
      } else
        vals.push_back(1);
    }

    rhs = std::floor(rhs + 0.5);

    cutpool.addCut(mipsolver, inds.data(), vals.data(), inds.size(), rhs, true,
                   false, false);
  }

  if (runcliquesubsumption) {
    if (cliquehits.size() < cliques.size()) cliquehits.resize(cliques.size());

    for (std::vector<CliqueVar>& clique : data.cliques) {
      HighsInt nremoved = runCliqueSubsumption(globaldom, clique);

      if (clique.empty()) continue;
      if (nremoved != 0) doAddClique(clique.data(), clique.size(), false, -1);
    }
  }
}

std::vector<std::vector<HighsCliqueTable::CliqueVar>>
HighsCliqueTable::separateCliques(const std::vector<double>& sol,
                                  const HighsDomain& globaldom,
                                  double feastol) {
  BronKerboschData data(sol);
  data.feastol = feastol;

  HighsInt numcols = globaldom.col_lower_.size();
  assert(int(numcliquesvar.size()) == 2 * numcols);
  for (HighsInt i = 0; i != numcols; ++i) {
    if (colsubstituted[i]) continue;

    if (numcliquesvar[CliqueVar(i, 0).index()] != 0 &&
        CliqueVar(i, 0).weight(sol) > feastol)
      data.P.emplace_back(i, 0);
    if (numcliquesvar[CliqueVar(i, 1).index()] != 0 &&
        CliqueVar(i, 1).weight(sol) > feastol)
      data.P.emplace_back(i, 1);
  }

  bronKerboschRecurse(data, data.P.size(), nullptr, 0);

  return std::move(data.cliques);
}

void HighsCliqueTable::addImplications(HighsDomain& domain, HighsInt col,
                                       HighsInt val) {
  CliqueVar v(col, val);

  while (colsubstituted[v.col]) {
    assert((HighsInt)substitutions.size() > colsubstituted[v.col] - 1);
    Substitution subst = substitutions[colsubstituted[v.col] - 1];
    v = v.val == 1 ? subst.replace : subst.replace.complement();
    if (v.val == 1) {
      if (domain.col_lower_[v.col] == 1.0) continue;

      domain.changeBound(HighsBoundType::kLower, v.col, 1.0,
                         HighsDomain::Reason::cliqueTable(col, val));
      if (domain.infeasible()) return;
    } else {
      if (domain.col_upper_[v.col] == 0.0) continue;

      domain.changeBound(HighsBoundType::kUpper, v.col, 0.0,
                         HighsDomain::Reason::cliqueTable(col, val));
      if (domain.infeasible()) return;
    }
  }

  auto doFixings = [&](HighsInt cliqueid) {
    HighsInt start = cliques[cliqueid].start;
    HighsInt end = cliques[cliqueid].end;

    for (HighsInt i = start; i != end; ++i) {
      if (cliqueentries[i].col == v.col) continue;

      if (cliqueentries[i].val == 1) {
        if (domain.col_upper_[cliqueentries[i].col] == 0.0) continue;

        domain.changeBound(HighsBoundType::kUpper, cliqueentries[i].col, 0.0,
                           HighsDomain::Reason::cliqueTable(col, val));
        if (domain.infeasible()) return true;
      } else {
        if (domain.col_lower_[cliqueentries[i].col] == 1.0) continue;

        domain.changeBound(HighsBoundType::kLower, cliqueentries[i].col, 1.0,
                           HighsDomain::Reason::cliqueTable(col, val));
        if (domain.infeasible()) return true;
      }
    }

    return false;
  };

  CliqueSet vClqs(this, v);
  HighsInt node = vClqs.first();
  while (node != -1) {
    if (doFixings(cliquesets[node].cliqueid)) return;
    node = vClqs.successor(node);
  }

  CliqueSet vClqsSizeTwo(this, v, true);
  node = vClqsSizeTwo.first();
  while (node != -1) {
    if (doFixings(cliquesets[node].cliqueid)) return;
    node = vClqsSizeTwo.successor(node);
  }
}

void HighsCliqueTable::cleanupFixed(HighsDomain& globaldom) {
  HighsInt numcol = globaldom.col_upper_.size();
  HighsInt oldnfixings = nfixings;
  for (HighsInt i = 0; i != numcol; ++i) {
    if (colDeleted[i] || globaldom.col_lower_[i] != globaldom.col_upper_[i])
      continue;
    if (globaldom.col_lower_[i] != 1.0 && globaldom.col_lower_[i] != 0.0)
      continue;

    HighsInt fixval = (HighsInt)globaldom.col_lower_[i];
    CliqueVar v(i, 1 - fixval);

    vertexInfeasible(globaldom, v.col, v.val);
    if (globaldom.infeasible()) return;
  }

  if (nfixings != oldnfixings) propagateAndCleanup(globaldom);
}

HighsInt HighsCliqueTable::getNumImplications(HighsInt col) {
  // first count all cliques as one implication, so that cliques of size two
  // are accounted for already
  HighsInt numimplics = numcliquesvar[CliqueVar(col, 0).index()] +
                        numcliquesvar[CliqueVar(col, 1).index()];

  // now loop over cliques larger than size two and add the cliquelength - 1 as
  // implication but subtract 1 more as each clique is already counted as one
  // implication
  CliqueSet clqsVal0(this, CliqueVar(col, 0));
  HighsInt node = clqsVal0.first();
  while (node != -1) {
    HighsInt nimplics = cliques[cliquesets[node].cliqueid].end -
                        cliques[cliquesets[node].cliqueid].start - 1;
    nimplics *= (1 + cliques[cliquesets[node].cliqueid].equality);
    numimplics += nimplics - 1;
    node = clqsVal0.successor(node);
  }

  CliqueSet clqsVal1(this, CliqueVar(col, 1));
  node = clqsVal1.first();
  while (node != -1) {
    HighsInt nimplics = cliques[cliquesets[node].cliqueid].end -
                        cliques[cliquesets[node].cliqueid].start - 1;
    nimplics *= (1 + cliques[cliquesets[node].cliqueid].equality);
    numimplics += nimplics - 1;
    node = clqsVal1.successor(node);
  }

  return numimplics;
}

HighsInt HighsCliqueTable::getNumImplications(HighsInt col, bool val) {
  HighsInt numimplics = numcliquesvar[CliqueVar(col, val).index()];

  CliqueSet clqSet(this, CliqueVar(col, val));
  HighsInt node = clqSet.first();
  while (node != -1) {
    HighsInt nimplics = cliques[cliquesets[node].cliqueid].end -
                        cliques[cliquesets[node].cliqueid].start - 1;
    nimplics *= (1 + cliques[cliquesets[node].cliqueid].equality);
    numimplics += nimplics - 1;
    node = clqSet.successor(node);
  }

  return numimplics;
}

void HighsCliqueTable::runCliqueMerging(HighsDomain& globaldomain,
                                        std::vector<CliqueVar>& clique,
                                        bool equation) {
  CliqueVar extensionstart;
  HighsInt numcliques = kHighsIInf;
  iscandidate.resize(numcliquesvar.size());
  HighsInt initialCliqueSize = clique.size();
  for (HighsInt i = 0; i != initialCliqueSize; ++i) {
    if (globaldomain.isFixed(cliqueentries[i].col)) continue;

    if (numcliquesvar[clique[i].index()] < numcliques) {
      numcliques = numcliquesvar[clique[i].index()];
      extensionstart = clique[i];
    }
  }

  if (numcliques == kHighsIInf) {
    clique.clear();
    return;
  }

  for (HighsInt i = 0; i != initialCliqueSize; ++i)
    iscandidate[clique[i].index()] = true;

  HighsInt node;
  auto addCands = [&]() {
    HighsInt start = cliques[cliquesets[node].cliqueid].start;
    HighsInt end = cliques[cliquesets[node].cliqueid].end;

    for (HighsInt i = start; i != end; ++i) {
      if (iscandidate[cliqueentries[i].index()] ||
          globaldomain.isFixed(cliqueentries[i].col))
        continue;

      iscandidate[cliqueentries[i].index()] = true;
      clique.push_back(cliqueentries[i]);
    }
  };

  CliqueSet clqSet(this, extensionstart);
  node = clqSet.first();
  while (node != -1) {
    addCands();
    node = clqSet.successor(node);
  }

  CliqueSet clqSetSizeTwo(this, extensionstart, true);
  node = clqSetSizeTwo.first();
  while (node != -1) {
    addCands();
    node = clqSetSizeTwo.successor(node);
  }

  HighsInt sizeWithCandidates = clique.size();
  for (HighsInt i = 0; i != sizeWithCandidates; ++i)
    iscandidate[clique[i].index()] = false;

  for (HighsInt i = 0;
       i != initialCliqueSize && initialCliqueSize < (HighsInt)clique.size();
       ++i) {
    if (clique[i] == extensionstart) continue;

    HighsInt newSize =
        initialCliqueSize +
        shrinkToNeighborhood(clique[i], clique.data() + initialCliqueSize,
                             clique.size() - initialCliqueSize);
    clique.erase(clique.begin() + newSize, clique.end());
  }

  if (initialCliqueSize < (HighsInt)clique.size()) {
    // todo, shuffle extension vars?
    randgen.shuffle(clique.data() + initialCliqueSize,
                    clique.size() - initialCliqueSize);
    HighsInt i = initialCliqueSize;
    while (i < (HighsInt)clique.size()) {
      CliqueVar extvar = clique[i];
      i += 1;

      HighsInt newSize = i + shrinkToNeighborhood(extvar, clique.data() + i,
                                                  clique.size() - i);
      clique.erase(clique.begin() + newSize, clique.end());
    }
  }

  if (equation) {
    for (HighsInt i = initialCliqueSize; i < (HighsInt)clique.size(); ++i)
      vertexInfeasible(globaldomain, clique[i].col, clique[i].val);
  } else {
    runCliqueSubsumption(globaldomain, clique);

    if (!clique.empty()) {
      clique.erase(
          std::remove_if(clique.begin(), clique.end(),
                         [&](CliqueVar v) {
                           return globaldomain.isFixed(v.col) &&
                                  int(globaldomain.col_lower_[v.col]) ==
                                      (1 - v.val);
                         }),
          clique.end());
    }
  }

  processInfeasibleVertices(globaldomain);
}

void HighsCliqueTable::runCliqueMerging(HighsDomain& globaldomain) {
  std::vector<CliqueVar> extensionvars;
  iscandidate.resize(numcliquesvar.size());

  if (cliquehits.size() < cliques.size()) cliquehits.resize(cliques.size());

  HighsInt numcliqueslots = cliques.size();
  const HighsInt maxNewEntries = numEntries + globaldomain.numModelNonzeros();
  bool haveNonModelCliquesToMerge = false;
  for (HighsInt k = 0; k != numcliqueslots; ++k) {
    if (cliques[k].start == -1) continue;
    if (!cliques[k].equality && cliques[k].origin == kHighsIInf) continue;
    if (cliques[k].origin == -1) {
      haveNonModelCliquesToMerge = true;
      continue;
    }
    HighsInt numclqvars = cliques[k].end - cliques[k].start;
    assert(numclqvars != 0);
    if (numclqvars == 0) continue;

    CliqueVar* clqvars = &cliqueentries[cliques[k].start];

    CliqueVar extensionstart = clqvars[0];
    HighsInt numcliques = numcliquesvar[clqvars[0].index()];
    for (HighsInt i = 1; i != numclqvars; ++i) {
      if (numcliquesvar[clqvars[i].index()] < numcliques) {
        numcliques = numcliquesvar[clqvars[i].index()];
        extensionstart = clqvars[i];
      }
    }

    for (HighsInt i = 0; i != numclqvars; ++i)
      iscandidate[clqvars[i].index()] = true;

    HighsInt node;
    auto addCands = [&]() {
      HighsInt start = cliques[cliquesets[node].cliqueid].start;
      HighsInt end = cliques[cliquesets[node].cliqueid].end;

      for (HighsInt i = start; i != end; ++i) {
        if (iscandidate[cliqueentries[i].index()] ||
            globaldomain.isFixed(cliqueentries[i].col))
          continue;

        iscandidate[cliqueentries[i].index()] = true;
        extensionvars.push_back(cliqueentries[i]);
      }
    };

    CliqueSet clqSet(this, extensionstart);
    node = clqSet.first();
    while (node != -1) {
      addCands();
      node = clqSet.successor(node);
    }

    CliqueSet clqSetSizeTwo(this, extensionstart, true);
    node = clqSetSizeTwo.first();
    while (node != -1) {
      addCands();
      node = clqSetSizeTwo.successor(node);
    }

    for (HighsInt i = 0; i != numclqvars; ++i)
      iscandidate[clqvars[i].index()] = false;
    for (CliqueVar v : extensionvars) iscandidate[v.index()] = false;

    for (HighsInt i = 0; i != numclqvars && !extensionvars.empty(); ++i) {
      if (clqvars[i] == extensionstart) continue;

      HighsInt newSize = shrinkToNeighborhood(clqvars[i], extensionvars.data(),
                                              extensionvars.size());
      extensionvars.erase(extensionvars.begin() + newSize, extensionvars.end());
    }

    if (!extensionvars.empty()) {
      // todo, shuffle extension vars?
      randgen.shuffle(extensionvars.data(), extensionvars.size());
      size_t i = 0;
      while (i < extensionvars.size()) {
        CliqueVar extvar = extensionvars[i];
        i += 1;

        HighsInt newSize =
            i + shrinkToNeighborhood(extvar, extensionvars.data() + i,
                                     extensionvars.size() - i);
        extensionvars.erase(extensionvars.begin() + newSize,
                            extensionvars.end());
      }
    }

    if (cliques[k].equality) {
      for (CliqueVar v : extensionvars)
        vertexInfeasible(globaldomain, v.col, v.val);
    } else {
      HighsInt originrow = cliques[k].origin;
      cliques[k].origin = kHighsIInf;

      HighsInt numExtensions = extensionvars.size();
      extensionvars.insert(extensionvars.end(),
                           cliqueentries.begin() + cliques[k].start,
                           cliqueentries.begin() + cliques[k].end);
      extensionvars.erase(
          std::remove_if(
              extensionvars.begin() + numExtensions, extensionvars.end(),
              [&](CliqueVar clqvar) { return colDeleted[clqvar.col]; }),
          extensionvars.end());
      removeClique(k);

      for (CliqueVar v : extensionvars) {
        CliqueSet clqSet(this, v);
        node = clqSet.first();
        while (node != -1) {
          HighsInt cliqueid = cliquesets[node].cliqueid;
          if (cliquehits[cliqueid] == 0) cliquehitinds.push_back(cliqueid);

          ++cliquehits[cliqueid];

          node = clqSet.successor(node);
        }

        CliqueSet clqSetSizeTwo(this, v, true);
        node = clqSetSizeTwo.first();
        while (node != -1) {
          HighsInt cliqueid = cliquesets[node].cliqueid;
          if (cliquehits[cliqueid] == 0) cliquehitinds.push_back(cliqueid);

          ++cliquehits[cliqueid];

          node = clqSetSizeTwo.successor(node);
        }
      }

      bool redundant = false;
      HighsInt numRemoved = 0;
      HighsInt dominatingOrigin = kHighsIInf;
      for (HighsInt cliqueid : cliquehitinds) {
        HighsInt hits = cliquehits[cliqueid];
        cliquehits[cliqueid] = 0;

        if (hits == extensionvars.size()) {
          redundant = true;
          if (cliques[cliqueid].origin != kHighsIInf &&
              cliques[cliqueid].origin != -1)
            dominatingOrigin = cliques[cliqueid].origin;
        } else if (cliques[cliqueid].end - cliques[cliqueid].start -
                       cliques[cliqueid].numZeroFixed ==
                   hits) {
          if (cliques[cliqueid].equality) {
            for (CliqueVar v : extensionvars) {
              bool sizeTwo =
                  cliques[cliqueid].end - cliques[cliqueid].start == 2;
              CliqueSet clqSet(this, v, sizeTwo);
              if (!clqSet.find(cliqueid).second) infeasvertexstack.push_back(v);
            }
          } else {
            ++numRemoved;
            removeClique(cliqueid);
          }
        }
      }

      cliquehitinds.clear();

      if (!redundant) {
        for (HighsInt i = 0; i < numExtensions; ++i)
          cliqueextensions.emplace_back(originrow, extensionvars[i]);

        extensionvars.erase(
            std::remove_if(extensionvars.begin(), extensionvars.end(),
                           [&](CliqueVar v) {
                             return globaldomain.isFixed(v.col) &&
                                    int(globaldomain.col_lower_[v.col]) ==
                                        (1 - v.val);
                           }),
            extensionvars.end());

        if (extensionvars.size() > 1)
          doAddClique(extensionvars.data(), extensionvars.size(), false,
                      originrow);
      } else {
        // the extended clique is redundant, check if the row can be removed
        if (dominatingOrigin != kHighsIInf)
          deletedrows.push_back(originrow);
        else {
          // this clique is redundant in the cliquetable but its row is not
          // necessarily. Also there might be rows that have been deleted due to
          // being dominated by this row after adding the lifted entries so they
          // must be added to the cliqueextension vector
          for (HighsInt i = 0; i < numExtensions; ++i)
            cliqueextensions.emplace_back(originrow, extensionvars[i]);
        }
      }
    }

    extensionvars.clear();
    processInfeasibleVertices(globaldomain);

    if (numEntries >= maxNewEntries) break;
    // printf("nonzeroDelta: %d, maxNonzeroDelta: %d\n", nonzeroDelta,
    // maxNonzeroDelta);
  }

  if (haveNonModelCliquesToMerge) {
    for (HighsInt k = 0; k != numcliqueslots; ++k) {
      if (cliques[k].start == -1) continue;
      if (cliques[k].origin != -1) continue;
      // if (cliques[k].end - cliques[k].start <= 1000) continue;

      // printf("numEntries before: %d\n", numEntries);
      extensionvars.clear();
      extensionvars.insert(extensionvars.end(),
                           cliqueentries.begin() + cliques[k].start,
                           cliqueentries.begin() + cliques[k].end);
      removeClique(k);
      runCliqueMerging(globaldomain, extensionvars);

      if (extensionvars.size() > 1)
        doAddClique(extensionvars.data(), extensionvars.size());
    }
  }
}

void HighsCliqueTable::rebuild(
    HighsInt ncols, const presolve::HighsPostsolveStack& postSolveStack,
    const HighsDomain& globaldomain,
    const std::vector<HighsInt>& orig2reducedcol,
    const std::vector<HighsInt>& orig2reducedrow) {
  HighsCliqueTable newCliqueTable(ncols);
  newCliqueTable.setPresolveFlag(inPresolve);
  newCliqueTable.setMinEntriesForParallelism(minEntriesForParallelism);
  HighsInt ncliques = cliques.size();
  for (HighsInt i = 0; i != ncliques; ++i) {
    if (cliques[i].start == -1) continue;

    for (HighsInt k = cliques[i].start; k != cliques[i].end; ++k) {
      HighsInt col = orig2reducedcol[cliqueentries[k].col];

      if (col == -1 || !globaldomain.isBinary(col) ||
          !postSolveStack.isColLinearlyTransformable(col))
        cliqueentries[k].col = kHighsIInf;
      else
        cliqueentries[k].col = col;
    }

    auto newend =
        std::remove_if(cliqueentries.begin() + cliques[i].start,
                       cliqueentries.begin() + cliques[i].end,
                       [](CliqueVar v) { return v.col == kHighsIInf; });
    HighsInt numvars = newend - (cliqueentries.begin() + cliques[i].start);
    // since we do not know how variables in the clique that have been deleted
    // are replaced (i.e. are they fixed to 0 or 1, or substituted) we relax
    // them out which means the equality status needs to be set to false
    if (numvars <= 1) continue;

    HighsInt origin = cliques[i].origin != kHighsIInf ? -1 : kHighsIInf;
    newCliqueTable.doAddClique(&cliqueentries[cliques[i].start], numvars, false,
                               origin);
  }

  *this = std::move(newCliqueTable);
}

void HighsCliqueTable::buildFrom(const HighsLp* origModel,
                                 const HighsCliqueTable& init) {
  assert(init.colsubstituted.size() == colsubstituted.size());
  HighsInt ncols = init.colsubstituted.size();
  HighsCliqueTable newCliqueTable(ncols);
  newCliqueTable.setPresolveFlag(inPresolve);
  newCliqueTable.setPresolveFlag(minEntriesForParallelism);
  HighsInt ncliques = init.cliques.size();
  std::vector<CliqueVar> clqBuffer;
  clqBuffer.reserve(2 * origModel->num_col_);
  for (HighsInt i = 0; i != ncliques; ++i) {
    if (init.cliques[i].start == -1) continue;

    HighsInt numvars = init.cliques[i].end - init.cliques[i].start;

    if (numvars - init.cliques[i].numZeroFixed <= 1) continue;

    clqBuffer.assign(init.cliqueentries.begin() + init.cliques[i].start,
                     init.cliqueentries.begin() + init.cliques[i].end);
    clqBuffer.erase(std::remove_if(clqBuffer.begin(), clqBuffer.end(),
                                   [origModel](CliqueVar v) {
                                     return origModel->col_lower_[v.col] !=
                                                0.0 ||
                                            origModel->col_upper_[v.col] != 1.0;
                                   }),
                    clqBuffer.end());
    if (clqBuffer.size() <= 1) continue;

    HighsInt origin = init.cliques[i].origin != kHighsIInf ? -1 : kHighsIInf;
    newCliqueTable.doAddClique(clqBuffer.data(), clqBuffer.size(), false,
                               origin);
  }

  newCliqueTable.colsubstituted = init.colsubstituted;
  newCliqueTable.substitutions = init.substitutions;
  *this = std::move(newCliqueTable);
}
