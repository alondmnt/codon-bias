from itertools import starmap
from multiprocessing.pool import Pool
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from .stats import CodonCounter


class PairwiseScore(object):
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def get_score(self, seq1, seq2):
        return self._calc_pair_score(
            self._calc_weights(seq1), self._calc_weights(seq2))

    def get_matrix(self, seqs, elementwise=False):
        self.weights = self._calc_weights(seqs)

        if not elementwise and hasattr(self, '_calc_matrix'):
            return self._calc_matrix()

        # the following uses self._calc_pair_score(), assuming that
        # the score is a symmetric distance with zeros on the diagonal
        n = len(seqs)
        if self.n_jobs == 1:
            sf = list(starmap(self._calc_matrix_element, zip(*np.triu_indices(n, k=1))))
        with Pool(self.n_jobs) as pool:
            sf = pool.starmap(self._calc_matrix_element, zip(*np.triu_indices(n, k=1)))

        return squareform(sf)

    def _calc_matrix_element(self, i, j):
        """ fallback function for when self._calc_matrix() is missing. """
        return self._calc_pair_score(self.weights[i], self.weights[j])

    def _calc_weights(self, seqs):
        raise('not implemented')

    def _calc_pair_score(self, w1, w2):
        raise('not implemented')


class CodonUsageFrequencySimilarity(PairwiseScore):
    def __init__(self, synonymous=False, genetic_code=1, ignore_stop=False, n_jobs=None):
        """ Diament, Pinter & Tuller, Nature Communications 2014. """
        super().__init__(n_jobs=n_jobs)
        self.synonymous = synonymous
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

    def _calc_weights(self, seqs):
        if isinstance(seqs, str):
            seqs = [seqs]
        counts = CodonCounter(seqs,
            sum_seqs=False, genetic_code=self.genetic_code,
            ignore_stop=self.ignore_stop)

        if not self.synonymous:
            return counts.get_codon_table(normed=True).T.values.astype(np.float32)

        weights = counts.get_aa_table(normed=True)
        # convert NaNs to a uniform distribution
        norm = weights.groupby('aa').size().to_frame('deg').join(weights)
        if type(weights) == pd.DataFrame:
            weights = weights.apply(lambda x: x.fillna(1 / norm['deg']))
        else:
            weights = weights.fillna(1 / norm['deg'])
        return weights.T.values.astype(np.float32)

    def _calc_pair_score(self, w1, w2):
        M = 0.5*(w1 + w2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sqrt(self._kld(w1, M) + self._kld(w2, M))

    def _kld(self, p, q):
        return np.nansum(np.log(p / q) * p, axis=0)

    def _calc_matrix(self):
        w1 = self.weights.T[:,:,None]
        w2 = self.weights.T[:,None,:]

        return self._calc_pair_score(w1, w2)
