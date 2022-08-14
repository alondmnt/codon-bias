import numpy as np

complement = {'A': 'T',
              'C': 'G',
              'G': 'C',
              'T': 'A'}


def reverse_complement(seq):
    return ''.join([complement[b] for b in seq[::-1]])


def geomean(weights, counts):
    nn = weights.index[np.isfinite(np.log(weights))]
    return np.exp((np.log(weights[nn]) * counts.reindex(nn)).sum() / counts.reindex(nn).sum())
