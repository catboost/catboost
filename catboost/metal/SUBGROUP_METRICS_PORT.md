# PFound subgroup metadata

Checkpoint `20260913T090442Z` is installed. It passes 5923 combined tests plus
16 subtests, 824 alternate native acceptance cases, 30 installed GPU paths
and six native CLI variants. Wheel SHA256 `8f15f439741ec7bb48f4715238e09417694c56036ce5755b68c7525a93b932ce`.

The shared CUDA fallback PFound metric calls TPFoundCalcer with query-local
subgroup IDs. It sorts by descending approximation, breaks ties by ascending
relevance, visits the original top-k positions and skips previously visited
subgroups without reducing pLook. Native Metal already supplies this query
metadata; the standalone frontend now carries it through the same metric.

CatBoostMetalRanker.fit accepts subgroup_id and eval_subgroup_id. Evaluation
tuples may append subgroup IDs after group weight. Pool inputs expose
get_subgroup_id_hash(), returning copied uint32 hashes in current row order,
including sliced and quantized Pools. Raw array tokens use CatBoost's existing
hash routine; IDs and weights supplied with a Pool remain owned by that Pool.

The private metric bridge passes stored hashes directly to the shared C++
evaluator. Rehashing stored IDs can merge distinct subgroups: decimal tokens
47220 and 111524 both hash to 1651280859. This behavior is covered explicitly.
Public catboost.utils.eval_metric retains its original token-hashing behavior.

Subgroups affect PFound metrics and model selection, and leave CUDA-derived
target generation unchanged. Both learn and validation hashes are copied,
validated, and included in snapshot identity. Classic YetiRank and
YetiRankPairwise loss histories, weighted evaluation, callbacks, best models
and exact replay use the same subgroup-aware tracker.

51 new cases cover four token types, quantization/slicing/copy behavior,
independent PFound equations, top-k duplicate handling, original query weights,
rehashing collisions, invalid hashes/metadata, seven ranking objectives,
learn-only evaluation, Pool/array equality, best models and snapshots.
125 focused ranking/lifecycle tests pass before the complete regression.
Thirty installed smoke paths include native and standalone subgroup PFound
for QueryRMSE, YetiRank and YetiRankPairwise. No CPU CatBoost fitting occurs.

The prior sparse checkpoint 20260913T085149Z remains intact with its timing
and memory arrays. Complete CUDA RNG, categorical/P4 ranking, grouped Ordered,
dense-ID scaling and the other documented parity work remain open.
