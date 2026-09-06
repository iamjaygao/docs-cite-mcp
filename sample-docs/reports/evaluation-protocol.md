# Evaluation Protocol

## Binding rule

Every reported metric must carry two fields: `candidate_pool` and
`metric_version`. Comparing numbers across pools is forbidden, because the pools
differ in both size and relevance distribution.

## Gain assignment

Relevance grades map to gains as E=1.0, S=0.1, C=0.01, I=0.0. An earlier version
of the scoring script had S and C transposed, which shifted results by up to
0.030 NDCG.

## Bootstrap

Config comparisons resample queries, not observations. Report the mean delta, a
95% interval, and a two-sided p-value. A delta whose interval spans zero is not
a result.
