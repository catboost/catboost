
The value of this metric can not be calculated. The metric that is written to [output data](../../../concepts/output-data.md) if {{ error-function__YetiRank }} is optimized depends on the range of all _N_ target values ($i \in [1; N]$) of the dataset:
- $target_{i} \in [0; 1]$ — {{ error-function__PFound }}
- $target_{i} \notin [0; 1]$ — {{ error-function__ndcg }}
