from collections import Iterable

import numpy as np
import pandas as pd

from .stats import CodonCounter


class ScalarScore(object):
    def __init__(self):
        pass

    def get_score(self, seq, slice=None):
        if not isinstance(seq, str):
            return np.array([self.get_score(s, slice=slice) for s in seq])

        if slice is not None:
            return self._calc_score(seq[slice])
        else:
            return self._calc_score(seq)

    def _calc_score(self, seq):
        raise Exception('not implemented')

    def _geomean(self, weights, counts):
        # TODO: move?
        nn = weights.index[np.isfinite(np.log(weights))]
        return np.exp((np.log(weights[nn]) * counts.reindex(nn)).sum() / counts.reindex(nn).sum())


class VectorScore(object):
    def __init__(self):
        pass

    def get_vector(self, seq, slice=None):
        if not isinstance(seq, str):
            return np.array([self.get_vector(s, slice=slice) for s in seq])

        if slice is not None:
            return self._calc_vector(seq[slice])
        else:
            return self._calc_vector(seq)

    def _calc_vector(self, seq):
        raise Exception('not implemented')

    def _get_codon_vector(self, seq):
        return [seq[i:i+3] for i in range(0, len(seq), 3)]


class CodonAdaptationIndex(ScalarScore, VectorScore):
    def __init__(self, ref_seq, genetic_code=1, ignore_stop=True):
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

        self.weights = CodonCounter(ref_seq, genetic_code=genetic_code)\
            .get_aa_table().groupby('aa').apply(lambda x: x / x.max())
        if ignore_stop:
            self.weights['*'] = np.nan
        self.weights = self.weights.droplevel('aa')

    def _calc_score(self, seq):
        counts = CodonCounter(seq, self.genetic_code).counts

        return self._geomean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.loc[self._get_codon_vector(seq)].values
