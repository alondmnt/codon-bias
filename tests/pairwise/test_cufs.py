"""Regression tests for CodonUsageFrequency post count_array migration.

CUFS now goes through CodonCounter.count_array on the hot path instead
of count(seqs).get_codon_table(). The score is bit-equivalent to a
manual KL-divergence computation on (count_array + pseudocount), and
the legacy count(seqs) entry point is no longer reached during scoring.
"""

from unittest import mock

import numpy as np

from codonbias.pairwise import CodonUsageFrequency


def _manual_endres_schindelin(p, q):
    """Reference KL-based distance: sqrt(KLD(p, M) + KLD(q, M)) with
    M = 0.5*(p+q). Matches CUFS._calc_pair_score's own definition; here
    purely as an independent re-implementation against which CUFS's
    output should match."""
    M = 0.5 * (p + q)
    kld_pm = np.nansum(np.log(p / M) * p)
    kld_qm = np.nansum(np.log(q / M) * q)
    return np.sqrt(kld_pm + kld_qm)


def test_get_score_matches_manual_kld_non_synonymous():
    cuf = CodonUsageFrequency(synonymous=False, pseudocount=1)
    s1, s2 = "ATGAAGCGTGAACTGGCTAAGTAA", "ATGAAGCGCGAACTGGCTAAGTAA"

    c1 = cuf.counter.count_array(s1) + 1
    c2 = cuf.counter.count_array(s2) + 1
    p = c1 / c1.sum()
    q = c2 / c2.sum()
    expected = _manual_endres_schindelin(p, q)

    np.testing.assert_allclose(cuf.get_score(s1, s2), expected, rtol=1e-6)


def test_get_score_matches_manual_kld_synonymous():
    cuf = CodonUsageFrequency(synonymous=True, pseudocount=1)
    s1, s2 = "ATGAAGCGTGAACTGGCTAAGTAA", "ATGAAGCGCGAACTGGCTAAGTAA"

    aa_groups = cuf.counter.aa_group
    c1 = cuf.counter.count_array(s1) + 1
    c2 = cuf.counter.count_array(s2) + 1
    aa_total_1 = np.bincount(aa_groups, weights=c1, minlength=int(aa_groups.max()) + 1)
    aa_total_2 = np.bincount(aa_groups, weights=c2, minlength=int(aa_groups.max()) + 1)
    p = c1 / aa_total_1[aa_groups]
    q = c2 / aa_total_2[aa_groups]
    expected = _manual_endres_schindelin(p, q)

    np.testing.assert_allclose(cuf.get_score(s1, s2), expected, rtol=1e-6)


def test_get_matrix_does_not_call_counter_count():
    """The point of the migration: count_array is the hot-path entry,
    not count(seqs) (which builds the pandas Series/DataFrame)."""
    cuf = CodonUsageFrequency(pseudocount=1)
    seqs = ["ATGAAGCGTGAA", "ATGAAGCGCGAA", "ATGAAACGTGAA"]

    with mock.patch.object(cuf.counter, "count") as count_method:
        cuf.get_matrix(seqs)

    count_method.assert_not_called()


def test_get_matrix_n_jobs_1_still_correct_after_migration():
    """Sanity: the n_jobs=1 fix from earlier still holds with the new
    weights path."""
    cuf_seq = CodonUsageFrequency(pseudocount=1, n_jobs=1)
    cuf_pool = CodonUsageFrequency(pseudocount=1, n_jobs=2)
    seqs = ["ATGAAGCGTGAA", "ATGAAGCGCGAA", "ATGAAACGTGAA"]

    np.testing.assert_allclose(cuf_seq.get_matrix(seqs), cuf_pool.get_matrix(seqs))


def test_get_score_kmer_2():
    """Smoke test: k_mer=2 path uses aa_group_kmer for synonymous and
    works through count_array too."""
    cuf = CodonUsageFrequency(synonymous=True, k_mer=2, pseudocount=1)
    s1 = "ATGAAGCGTGAACTGGCTAAGAAATAA"
    s2 = "ATGAAGCGCGAACTGGCTAAGAAATAA"

    score = cuf.get_score(s1, s2)
    assert np.isfinite(score)
    assert score >= 0
