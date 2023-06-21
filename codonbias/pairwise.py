from itertools import starmap
from multiprocessing.pool import Pool
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from .stats import CodonCounter


class PairwiseScore(object):
    """
    Abstract class for models that output a scalar for a pair of sequences,
    or a pairwise score matrix for a set of sequences. Inheriting classes
    may implement the computation of the score for a single pair in two
    steps: (1) a transformation of the sequence by `_calc_weights(seq)`;
    and (2) a computation of the score by `_calc_pair_score(w1, w2)`.
    The abstract class implements two wrapper methods that call the
    aforementioned internal implementations: `get_score(seq1, seq2)`,
    `get_matrix(seqs)`. The latter function assumes that the score is
    symmetric, and that the diagonal always contains zeros.

    In case that a dedicated implementation for whole matrix computation
    is implemented in `_calc_matrix(weights)`, this method will be
    preferred by the `get_matrix(seqs)` method. This can be, for example,
    an efficient vectorized implementation of the computation.

    Parameters
    ----------
    n_jobs : int, optional
        Number of processes to use for matrix computation. If None is
        provided then the number returned by os.cpu_count() is used, by
        default None
    """
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def get_score(self, seq1, seq2):
        """
        Computes the score between the two given sequences. 

        Parameters
        ----------
        seq1 : str
            DNA sequence.
        seq2 : str
            DNA sequence.

        Returns
        -------
        float
            Score for `seq1` and `seq2`.
        """
        return self._calc_pair_score(
            self._calc_weights(seq1), self._calc_weights(seq2))

    def get_matrix(self, seqs, elementwise=False):
        """
        Computes the all pair score matrix for the given sequences.

        Parameters
        ----------
        seqs : iterable of str
            Set of DNA sequences.
        elementwise : bool, optional
            When True matrix computation will be done element by element
            using multiple processes. This may be useful to decrease
            memory consumption, by default False

        Returns
        -------
        numpy.array
            Square matrix of scores for all pairs of the given sequences.
        """
        self.weights = self._calc_weights(seqs)

        if not elementwise and hasattr(self, '_calc_matrix'):
            return self._calc_matrix(self.weights)

        # the following uses self._calc_pair_score(), assuming that
        # the score is a symmetric distance with zeros on the diagonal
        n = len(seqs)
        if self.n_jobs == 1:
            sf = list(starmap(self._calc_matrix_element, zip(*np.triu_indices(n, k=1))))
        with Pool(self.n_jobs) as pool:
            sf = pool.starmap(self._calc_matrix_element, zip(*np.triu_indices(n, k=1)))

        return squareform(sf)

    def _calc_matrix_element(self, i, j):
        """ Fallback function for when self._calc_matrix() is missing. """
        return self._calc_pair_score(self.weights[i], self.weights[j])

    def _calc_weights(self, seqs):
        raise('not implemented')

    def _calc_pair_score(self, w1, w2):
        raise('not implemented')


class CodonUsageFrequency(PairwiseScore):
    """
    Codon Usage Frequency (CUFS, Diament, Pinter & Tuller, Nature
    Communications, 2014).

    This is a distance metric between pairs of sequences based on their
    distribution of codons. It employs a distance metric for probability
    distrbutions (Endres & Schindelin, 2003) that is based on KL
    divergence. The original implementation used the parameter
    `pseudocount`=0.

    Parameters
    ----------
    synonymous : bool, optional
        When True snynomous codon frequencies are normalized to sum to 1
        for each amino acid (synCUFS), by default False
    k_mer : int, optional
        Determines the length of the codon k-mer to base statistics on, by
        default 1
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default False
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies, by
        default 1
    n_jobs : _type_, optional
        Number of processes to use for matrix computation. If None is
        provided then the number returned by os.cpu_count() is used, by
        default None
    """
    def __init__(self, synonymous=False, k_mer=1, genetic_code=1,
                 ignore_stop=False, pseudocount=1, n_jobs=None):
        super().__init__(n_jobs=n_jobs)
        self.synonymous = synonymous
        self.counter = CodonCounter(sum_seqs=False,
                                    k_mer=k_mer, concat_index=True,
                                    genetic_code=genetic_code,
                                    ignore_stop=ignore_stop)
        self.pseudocount = pseudocount

    def _calc_weights(self, seqs):
        if isinstance(seqs, str):
            seqs = [seqs]
        counts = self.counter.count(seqs)

        if not self.synonymous:
            return counts.get_codon_table(
                normed=True, pseudocount=self.pseudocount)\
                .T.values.astype(np.float32)

        weights = counts.get_aa_table(
            normed=True, pseudocount=self.pseudocount)

        return weights.T.values.astype(np.float32)

    def _calc_pair_score(self, w1, w2):
        M = 0.5*(w1 + w2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sqrt(self._kld(w1, M) + self._kld(w2, M))

    def _kld(self, p, q):
        return np.nansum(np.log(p / q) * p, axis=0)

    def _calc_matrix(self, weights):
        w1 = weights.T[:,:,None]
        w2 = weights.T[:,None,:]

        return self._calc_pair_score(w1, w2)
