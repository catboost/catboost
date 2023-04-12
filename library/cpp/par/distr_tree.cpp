#include "distr_tree.h"

#include <util/random/random.h>

namespace NPar {
    struct TTreeConnectorCost {
        int Vec1, Vec2;
        float Cost;
    };

    struct TTreeConnectorCostCmp {
        bool operator()(const TTreeConnectorCost& a, const TTreeConnectorCost& b) const {
            return a.Cost < b.Cost;
        }
    };

    struct TDistrTree {
        int FinalNodeId;
        TVector<TDistrTree> Children;
        float MaxPing; // max ping within this group
        float Cost;    // max delivery time of Children[i]

        TDistrTree()
            : FinalNodeId(-1)
            , MaxPing(0)
            , Cost(0)
        {
        }
    };

    // total cost = send time + max ping time + max within group delivery time
    static float CalcTreeCost(const TDistrTree& a, float sendTime) {
        return (a.Children.ysize() - 1) * sendTime + a.MaxPing + a.Cost;
    }

    struct TDistrTreeCmpCost {
        float SendTime;
        TDistrTreeCmpCost(float sendTime)
            : SendTime(sendTime)
        {
        }
        bool operator()(const TDistrTree& a, const TDistrTree& b) const {
            return CalcTreeCost(a, SendTime) > CalcTreeCost(b, SendTime);
        }
    };

    struct TDistrTreeConstructor {
        TVector<int> Group2parent;
        TVector<TDistrTree> Group;

        int GetParent(int id) {
            int res = id, parent = id;
            while (parent != -1) {
                res = parent;
                parent = Group2parent[res];
                if (parent == res) {
                    Y_ASSERT(id == res);
                    break;
                }
            }
            Group2parent[id] = res;
            return res;
        }
        TDistrTreeConstructor(int nodeCount) {
            //    TVector<TTreeConnectorCost> usedLinks;
            Group2parent.resize(nodeCount, -1);
            Group.reserve(nodeCount * 2);
            Group.resize(nodeCount);
            for (int i = 0; i < nodeCount; ++i) {
                Group[i].Children.resize(1);
                Group[i].Children[0].FinalNodeId = i;
            }
        }
        TDistrTree& AddTree(int* newGroupIdPtr) {
            int newGroupId = Group.ysize();
            *newGroupIdPtr = newGroupId;
            Group.resize(newGroupId + 1);
            Group2parent.resize(newGroupId + 1, -1);
            return Group[newGroupId];
        }
    };

    static void ConstructTree(TDistrTree* res, const TVector<TTreeConnectorCost>& links, int nodeCount, float sendTime) //treeNodeCost)
    {
        if (links.empty() || nodeCount == 1) {
            res->FinalNodeId = 0;
            return;
        }
        TDistrTreeConstructor treeConstr(nodeCount);

        for (int i = 0; i < links.ysize(); ++i) {
            const TTreeConnectorCost& tc = links[i];
            int g1 = treeConstr.GetParent(tc.Vec1);
            int g2 = treeConstr.GetParent(tc.Vec2);
            if (g1 == g2)
                continue; // already connected

            const TDistrTree& gg1 = treeConstr.Group[g1];
            const TDistrTree& gg2 = treeConstr.Group[g2];
            Y_ASSERT(tc.Cost > gg1.MaxPing && tc.Cost > gg2.MaxPing);
            float flatConnectionCost =
                (gg1.Children.ysize() + gg2.Children.ysize() - 1) * sendTime +
                tc.Cost +
                Max(gg1.Cost, gg2.Cost);

            float g1total = CalcTreeCost(gg1, sendTime);
            float g2total = CalcTreeCost(gg2, sendTime);
            float newNodeCost = sendTime + tc.Cost + Max(g1total, g2total);

            int newGroupId;
            TDistrTree& newGroup = treeConstr.AddTree(&newGroupId);
            if (flatConnectionCost <= newNodeCost) {
                // make flat marge
                newGroup.Children = gg1.Children;
                newGroup.Children.insert(newGroup.Children.end(), gg2.Children.begin(), gg2.Children.end());
                std::sort(newGroup.Children.begin(), newGroup.Children.end(), TDistrTreeCmpCost(sendTime));
                newGroup.Cost = Max(gg1.Cost, gg2.Cost);
                newGroup.MaxPing = tc.Cost;
            } else {
                // make hierarchical merge
                newGroup.Children.push_back(gg1);
                newGroup.Children.push_back(gg2);
                std::sort(newGroup.Children.begin(), newGroup.Children.end(), TDistrTreeCmpCost(sendTime));
                newGroup.Cost = Max(g1total, g2total);
                newGroup.MaxPing = tc.Cost;
            }
            treeConstr.Group2parent[g1] = newGroupId;
            treeConstr.Group2parent[g2] = newGroupId;
        }
        *res = treeConstr.Group[treeConstr.GetParent(0)];
    }

    static void ConstructTreeLimitBranching(TDistrTree* res, const TVector<TTreeConnectorCost>& links, int nodeCount, int maxBranching) {
        if (links.empty() || nodeCount == 1) {
            res->FinalNodeId = 0;
            return;
        }
        TDistrTreeConstructor treeConstr(nodeCount);

        int groupCount = nodeCount;
        for (int level = 1; groupCount > 1; ++level) {
            int limit = maxBranching;
            if (groupCount < maxBranching * 2)
                limit = maxBranching * 2;
            for (int i = 0; i < links.ysize(); ++i) {
                const TTreeConnectorCost& tc = links[i];
                int g1 = treeConstr.GetParent(tc.Vec1);
                int g2 = treeConstr.GetParent(tc.Vec2);
                if (g1 == g2)
                    continue; // already connected

                const TDistrTree& gg1 = treeConstr.Group[g1];
                const TDistrTree& gg2 = treeConstr.Group[g2];
                int g1count = gg1.Cost == level ? gg1.Children.ysize() : 1;
                int g2count = gg2.Cost == level ? gg2.Children.ysize() : 1;
                if (g1count + g2count > limit)
                    continue;

                int newGroupId;
                TDistrTree& newGroup = treeConstr.AddTree(&newGroupId);
                if (gg1.Cost == level)
                    newGroup.Children.insert(newGroup.Children.end(), gg1.Children.begin(), gg1.Children.end());
                else
                    newGroup.Children.push_back(gg1);
                if (gg2.Cost == level)
                    newGroup.Children.insert(newGroup.Children.end(), gg2.Children.begin(), gg2.Children.end());
                else
                    newGroup.Children.push_back(gg2);
                newGroup.Cost = level;

                treeConstr.Group2parent[g1] = newGroupId;
                treeConstr.Group2parent[g2] = newGroupId;
                --groupCount;
            }
        }
        *res = treeConstr.Group[treeConstr.GetParent(0)];
    }

    const ui16 N_GROUP_START = 0xffff;
    const ui16 N_GROUP_END = 0xfffe;
    const ui16 N_GROUP_FIRST_CODE = 0xfffe;

    static void OptimizeTreeEncoding(TVector<ui16>* res) {
        TVector<ui16>& vec = *res;
        for (int i = 0; i < vec.ysize();) {
            if (vec[i] == N_GROUP_START && vec[i + 1] == N_GROUP_END)
                vec.erase(vec.begin() + i, vec.begin() + i + 2);
            else if (vec[i] == N_GROUP_START && vec[i + 2] == N_GROUP_END) {
                vec[i] = vec[i + 1];
                vec.erase(vec.begin() + i + 1, vec.begin() + i + 3);
                ++i;
            } else
                ++i;
        }
        if (vec[0] == N_GROUP_START && vec.back() == N_GROUP_END) {
            vec.erase(vec.begin());
            vec.erase(vec.end() - 1);
        }
    }

    static void EncodeTreeImpl(const TDistrTree& t, TVector<ui16>* res) {
        if (t.Children.empty()) {
            Y_ASSERT(t.FinalNodeId < N_GROUP_FIRST_CODE);
            res->push_back(static_cast<ui16>(t.FinalNodeId));
        } else {
            if (t.Children.ysize() == 1)
                EncodeTreeImpl(t.Children[0], res);
            else {
                res->push_back(N_GROUP_START);
                for (int i = 0; i < t.Children.ysize(); ++i)
                    EncodeTreeImpl(t.Children[i], res);
                res->push_back(N_GROUP_END);
            }
        }
    }

    static void EncodeTree(const TDistrTree& t, TVector<ui16>* res) {
        res->resize(0);
        EncodeTreeImpl(t, res);
        OptimizeTreeEncoding(res);
    }

    void BuildDistributionTree(TDistributionTreesData* res,
                               const TArray2D<TVector<float>>& delayMatrixData) {
        Y_ASSERT(delayMatrixData.GetXSize() == delayMatrixData.GetYSize());
        int searcherCount = delayMatrixData.GetXSize();

        // write median to the delay matrix
        TArray2D<float> delayMatrix;
        delayMatrix.SetSizes(searcherCount, searcherCount);
        delayMatrix.FillZero();
        for (int srcHostId = 0; srcHostId < searcherCount; ++srcHostId) {
            for (int dstHostId = 0; dstHostId < searcherCount; ++dstHostId) {
                TVector<float> delays = delayMatrixData[srcHostId][dstHostId];
                if (delays.empty()) {
                    // out of luck, no delay data
                    delayMatrix[srcHostId][dstHostId] = 0;
                    continue;
                }
                std::sort(delays.begin(), delays.end());
                delayMatrix[srcHostId][dstHostId] = delays[delays.ysize() / 2];
            }
        }
        TVector<TTreeConnectorCost> links;
        links.reserve(searcherCount * searcherCount / 2);
        for (int srcHostId = 0; srcHostId < searcherCount; ++srcHostId) {
            for (int dstHostId = srcHostId + 1; dstHostId < searcherCount; ++dstHostId) {
                float delay12 = delayMatrix[srcHostId][dstHostId];
                float delay21 = delayMatrix[dstHostId][srcHostId];
                if (delay12 == 0)
                    delay12 = delay21;
                if (delay21 == 0)
                    delay21 = delay12;
                TTreeConnectorCost tc;
                tc.Cost = delay12 + delay21;
                tc.Vec1 = srcHostId;
                tc.Vec2 = dstHostId;
                if (tc.Cost == 0)
                    tc.Cost = 1e20f;
                links.push_back(tc);
            }
        }

        std::sort(links.begin(), links.end(), TTreeConnectorCostCmp());

        TDistrTree tree100K, tree1M, tree10M, treeBranch6;
        ConstructTree(&tree100K, links, searcherCount, 0.001f); // 1ms, 100K packets
        ConstructTree(&tree1M, links, searcherCount, 0.01f);
        ConstructTree(&tree10M, links, searcherCount, 0.1f);
        ConstructTreeLimitBranching(&treeBranch6, links, searcherCount, 6);
        EncodeTree(tree100K, &res->Tree100K);
        EncodeTree(tree1M, &res->Tree1M);
        EncodeTree(tree10M, &res->Tree10M);
        EncodeTree(treeBranch6, &res->TreeBranch6);
        //TOFStream f("c:/111mptest/tree.txt");
        //PrintTree(f, treeBranch6, 0, hostNames);
    }

    void GenerateSubtasks(const TVector<ui16>& src, TVector<TVector<ui16>>* subTasks) {
        for (int i = 0; i < src.ysize(); ++i) {
            TVector<ui16>& dst = subTasks->emplace_back();
            if (src[i] == N_GROUP_START) {
                // embedded group
                int depth = 1;
                for (++i; i < src.ysize(); ++i) {
                    ui16 val = src[i];
                    if (val == N_GROUP_END) {
                        if (--depth == 0)
                            break;
                    } else if (val == N_GROUP_START)
                        ++depth;
                    dst.push_back(src[i]);
                }
                Y_ASSERT(depth == 0);
            } else
                dst.push_back(src[i]);
        }
    }

    int SelectRandomHost(const TVector<ui16>& execPlan) {
        size_t ptr = RandomNumber(execPlan.size());
        while (execPlan[ptr] >= N_GROUP_FIRST_CODE)
            ptr = RandomNumber(execPlan.size());
        return execPlan[ptr];
    }

    void ProjectExecPlan(TVector<ui16>* res, const TVector<bool>& selectedComps) {
        for (int ptr = 0; ptr < res->ysize();) {
            ui16 comp = (*res)[ptr];
            if (comp >= N_GROUP_FIRST_CODE) {
                ++ptr;
                continue;
            }
            if (comp >= selectedComps.ysize() || selectedComps[comp] == false) {
                res->erase(res->begin() + ptr);
                while (ptr < res->ysize() && (*res)[ptr] == N_GROUP_END && (*res)[ptr - 1] == N_GROUP_START) {
                    res->erase(res->begin() + ptr - 1, res->begin() + ptr + 1);
                    --ptr;
                }
            } else
                ++ptr;
        }
    }

    void GetSelectedCompList(TVector<bool>* res, const TVector<ui16>& plan) {
        res->resize(0);
        for (int i = 0; i < plan.ysize(); ++i) {
            int n = plan[i];
            if (n >= N_GROUP_FIRST_CODE)
                continue;
            AddCompToSelectedList(res, n);
        }
    }
}
