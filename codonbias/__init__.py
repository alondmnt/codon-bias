"""
This package provides analysis tools for genomic sequences, focusing on
protein coding regions, translation efficiency and synonymous mutations.
These include implementations of popular models from the past four decades
of codon usage study, such as:

- Frequency of Optimal Codons (FOP)
- Relative Synonymous Codon Usage (RSCU)
- Codon Adaptation Index (CAI)
- Effective Number of Codons (ENC)
- tRNA Adaptation Index (tAI)
- Relative Codon Bias Score (RCBS)
- Directional Codon Bias Score (DCBS)
- Codon Usage Frequency Similarity (CUFS)

The package contains 4 submodules:

- codonbias.stats: Classes for codon statistics.
- codonbias.scores: Models / scores that operate on individual sequences
  independently.
- codonbias.pairwise: Models / scores that operate on pairs of sequences.
- codonbias.utils: Helper functions for the other submodules.

"""

__version__ = "0.1.0"
__author__ = 'Alon Diament'

import codonbias.utils
import codonbias.stats
import codonbias.scores
import codonbias.pairwise
