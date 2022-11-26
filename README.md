# cs6362-project

Final project for cs6362: advanced machine learning 

**Motivating question**: How do Variational Causal Networks (VCNs) approximation of the posterior over Direct Acyclic Graphs (DAGs) in causal structure learning compare to the boostrapped estimation via Direct LiNGAM in high dimensional settings?

VCN adapted from: https://github.com/yannadani/vcn_pytorch \
Boostrapped DirectLiNGAM adapted from: https://github.com/cdt15/lingam

**Data:** \
Data and verified ground truth source: 
1) https://pubmed.ncbi.nlm.nih.gov/15845847/ \
2) https://arxiv.org/abs/1805.03108
\
line 0-852: cd3_cd28 \
line 853-1754: icam2 \
line 1755-2665: aktinhib \
line 2666-3388: g0076 \
line 3389-4198: psitect \
line 4199-4997: u0126 \
line 4998-5845: ly \
line 5846-6758: pma \
line 6759-7465: b2camp
