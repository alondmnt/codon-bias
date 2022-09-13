"""
This package provides analysis tools for genomic sequences, focusing on
protein coding regions, translation efficiency and synonymous mutations.
These include implementations of popular models from the past four decades
of codon usage study, such as:

- Nucleotide and codon k-mer statistics (GC, GC3, CpG, etc.)
- Frequency of Optimal Codons (FOP)
- Relative Synonymous Codon Usage (RSCU)
- Codon Adaptation Index (CAI), including extensions:
    - Codon pair (and k-mers) adaptation
- Effective Number of Codons (ENC), including extensions:
    - Background correction
    - Improved estimation
    - Effective number of codon pairs (and k-mers) (ENcp)
- tRNA Adaptation Index (tAI)
- Codon Pair Bias (CPB/CPS)
- Relative Codon Bias Score (RCBS)
- Normalized Translational Efficiency (nTE)
- Directional Codon Bias Score (DCBS)
- Codon Usage Frequency Similarity (CUFS)

The package contains 4 submodules:

- codonbias.stats: Classes for basepair / codon statistics.
- codonbias.scores: Models / scores that operate on individual sequences
  independently.
- codonbias.pairwise: Models / scores that operate on pairs of sequences.
- codonbias.utils: Helper functions for the other submodules.

"""

__version__ = "0.2.0"
__author__ = 'Alon Diament'

import codonbias.utils
import codonbias.stats
import codonbias.scores
import codonbias.pairwise
