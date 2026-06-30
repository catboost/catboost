 ### models

#### Description

A list of models to blend.

**Possible values**

{{ python-type--list }} of {{ product }} models

**Default value**

{{ python--required }}

### weights

#### Description

A list of weights for the leaf values of each model. The length of this list must be equal to the number of blended models.

А list of weights equal to <q>1.0/N</q> for N blended models gives the average prediction. For example, the following list of weights gives the average prediction for four blended models:

```
[0.25,0.25,0.25,0.25]
```
**Possible values**

{{ python-type--list }} of numbers

**Default value**

None (leaf values weights are set to 1 for all models)

### ctr_merge_policy

#### Description

The counters merging policy. Possible values:
- {{ ECtrTableMergePolicy__FailIfCtrIntersects }} — Ensure that the models have zero intersecting counters.
- {{ ECtrTableMergePolicy__LeaveMostDiversifiedTable }} — Use the most diversified counters by the count of unique hash values.
- {{ ECtrTableMergePolicy__IntersectingCountersAverage }} — Use the average ctr counter values in the intersecting bins.
- {{ ECtrTableMergePolicy__KeepAllTables }} — Keep Counter and FeatureFreq ctr's from all models.

**Possible values**

{{ python-type--string }}

**Default value**

{{ ECtrTableMergePolicy__default }}
