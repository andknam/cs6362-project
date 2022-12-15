# cs6362-project

(_manuscript available upon request_)

Final project for cs6362: advanced machine learning 

**Motivating question**: How do Variational Causal Networks (VCNs) approximation of the posterior over Direct Acyclic Graphs (DAGs) in causal structure learning compare to the boostrapped estimation via Direct LiNGAM in high dimensional settings?

VCN adapted from: https://github.com/yannadani/vcn_pytorch (_instructions located under models/vcn_adapted_)

Boostrapped DirectLiNGAM adapted from: https://github.com/cdt15/lingam

**Data** (and verified ground truth source): 
 
1) https://pubmed.ncbi.nlm.nih.gov/15845847/ 
2) https://arxiv.org/abs/1805.03108

| Line # | Intervention |
| --- | ----------- |
| 0-852 | cd3_cd28 |
| 853-1754 | icam2 |
| 1755-2665 | aktinhib |
| 2666-3388 | g0076 |
| 3389-4198 | psitect |
| 4199-4997 | u0126 |
| 4998-5845 | ly |
| 5846-6758 | pma  |
| 6759-7465 | b2camp |

**Future work**:
ELBO gradient --> score function estimator variance reduction for more optimal learning within VCNs using control variates.

