"""Locks the n_jobs==1 sequential branch of PairwiseScore.get_matrix.

Until the dead-code fix, the n_jobs==1 starmap result was unconditionally
overwritten by a Pool.starmap call on the next line, so the sequential
path was never actually exercised — n_jobs==1 silently went through a
one-worker Pool. The output stayed correct (single-worker Pool gives
the same matrix), so an output-equality test wouldn't have caught it.
This test asserts the structural intent: n_jobs=1 must not enter Pool.
"""

from unittest import mock

from codonbias.pairwise import CodonUsageFrequency


def test_get_matrix_n_jobs_1_skips_pool():
    seqs = ["ATGAAGCGTGAA", "ATGAAGCGCGAA", "ATGAAACGTGAA"]

    with mock.patch("codonbias.pairwise.Pool") as pool_cls:
        CodonUsageFrequency(n_jobs=1).get_matrix(seqs, elementwise=True)

    pool_cls.assert_not_called()
