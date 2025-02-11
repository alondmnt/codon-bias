# codon-bias

This package provides codon usage bias (CUB) analysis tools for genomic sequences, focusing on protein coding regions, translation efficiency and synonymous mutations. These include implementations of popular models from the past four decades of codon usage study, such as:

- [Nucleotide and codon k-mer statistics (GC, GC3, CpG, etc.)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#module-codonbias.stats)
- [Frequency of Optimal Codons (FOP)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.FrequencyOfOptimalCodons)
- [Relative Synonymous Codon Usage (RSCU)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.RelativeSynonymousCodonUsage)
- [Codon Adaptation Index (CAI)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.CodonAdaptationIndex), including extensions:
    - Codon pair (and k-mers) adaptation
- [Effective Number of Codons (ENC)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.EffectiveNumberOfCodons), including extensions:
    - Background correction
    - Improved estimation
    - Effective number of codon pairs (and k-mers) (ENcp)
- [tRNA Adaptation Index (tAI)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.TrnaAdaptationIndex)
    - Download tRNA gene copy numbers from [GtRNAdb](http://gtrnadb.ucsc.edu/)
    - Train tAI model parameters (s-values) using expression levels
- [Codon Pair Bias (CPB/CPS)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.CodonPairBias)
- [Relative Codon Bias Score (RCBS)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.RelativeCodonBiasScore)
- [Normalized Translational Efficiency (nTE)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.NormalizedTranslationalEfficiency)
- [Directional Codon Bias Score (DCBS)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.scores.RelativeCodonBiasScore)
- [Codon Usage Frequency Similarity (CUFS)](https://codon-bias.readthedocs.io/en/latest/codonbias.html#codonbias.pairwise.CodonUsageFrequency)

This package also includes tools for sequence optimization based on these codon usage models, and generators of random sequence permutations that can be used to compute empirical p-values and z-scores.

## installation

```
pip install codon-bias
```

## documentation

Read on [Read the Docs](https://codon-bias.readthedocs.org).

## cite

Diament, A. (2022). codon-bias (python package). https://doi.org/10.5281/zenodo.8039451

## contributing

Contributions of additional models to the package are welcome! Please familiarize yourself with the existing classes, and try to conform to their style.
