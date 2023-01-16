
The following loss functions use Pairwise scoring:

- {{ error-function__YetiRankPairwise }}
- {{ error-function__PairLogitPairwise }}
- {{ error-function__QueryCrossEntropy }}

Pairwise scoring is slightly different from regular training on pairs, since pairs are generated only internally during the training for the corresponding metrics. One-hot encoding is not available for these loss functions.
