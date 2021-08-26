
The default value depends on the processing unit type and other parameters:

- {{ calcer_type__cpu }}: 254
- {{ calcer_type__gpu }} in {{ error-function__PairLogitPairwise }} and {{ error-function__YetiRankPairwise }} modes: 32
- {{ calcer_type__gpu }} in all other modes: 128
