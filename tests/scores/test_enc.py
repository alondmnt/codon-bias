import pytest
import numpy as np
import pandas as pd
import time
from numpy.testing import assert_allclose

import codonbias.scores
from codonbias.scores import EffectiveNumberOfCodons


@pytest.fixture
def enc_default():
    """Provides a default EffectiveNumberOfCodons instance."""
    return EffectiveNumberOfCodons()


@pytest.fixture
def random_seq_gen():
    """Factory fixture to generate random DNA sequences of a given length."""

    def _generate(length, seed=None):
        if seed is not None:
            np.random.seed(seed)
        bases = ['A', 'C', 'G', 'T']
        return ''.join(np.random.choice(bases, size=length))

    return _generate


# --- FUNCTIONAL TESTS ---

def test_enc_basic_logic(enc_default):
    """Verifies fundamental scoring for standard, biased, and edge cases."""
    # Standard multi-codon sequence
    assert_allclose(enc_default.get_score("ATGCGTACG"), 59.792271, rtol=1e-5)

    # Empty string / NaN handling
    assert np.isnan(enc_default.get_score(""))

    # Extreme bias (handled by pseudocount logic)
    assert_allclose(enc_default.get_score("ATGATGATGATG"), 61.0, rtol=1e-5)

    # Handling non-standard characters (should not crash)
    assert np.isfinite(enc_default.get_score("ATGCGNATGCGT"))


def test_enc_multiple_input_types(enc_default):
    """Verifies array-like inputs return numpy arrays."""
    seqs = ["ATGCGTACG", "ATGATGATG"]
    scores = enc_default.get_score(seqs)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2


@pytest.mark.parametrize("robust", [True, False])
@pytest.mark.parametrize("mean", ["weighted", "unweighted"])
def test_enc_parameters(robust, mean):
    """Checks all logic branches for robustness and mean calculation."""
    enc = EffectiveNumberOfCodons(robust=robust, mean=mean)
    seq = "ATGCGTACGACGTGA"
    score = enc.get_score(seq)
    assert np.isfinite(score)
    assert score > 0


def test_cython_python_equivalence(enc_default, random_seq_gen):
    """
    Verifies that the high-performance Cython path and the
    original Pandas path produce identical numerical results.
    """
    # Test across multiple lengths to ensure buffer management is correct
    for length in [30, 300, 1500]:
        seq = random_seq_gen(length, seed=length)

        # Force Pure Python (Pandas) path
        codonbias.scores.HAS_CYTHON = False
        score_py = enc_default.get_score(seq)

        # Force Optimized Cython path
        codonbias.scores.HAS_CYTHON = True
        score_cy = enc_default.get_score(seq)

        assert_allclose(score_cy, score_py, rtol=1e-7,
                        err_msg=f"Mismatch at length {length}")


def test_enc_dataframe_regression(enc_default, dataframe_regression):
    """
    Explicit regression test with 1000 sequences of varying ENC degrees.
    Generates sequences with different GC contents to ensure a range
    of scores (high and low bias) are tested and matched.
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    bases = np.array(['A', 'C', 'G', 'T'])

    all_scores = []

    # Generate 1000 sequences with varying bias
    for i in range(1000):
        # We vary the probability distribution to get different ENC results
        # Some iterations will be very biased, others will be uniform
        bias_factor = (i % 10) / 10.0  # Cycle through bias levels
        p = np.array([0.25, 0.25, 0.25, 0.25])

        # Shift probability to create high/low ENC scenarios
        if i % 2 == 0:
            p = np.array([0.1 + 0.4 * bias_factor, 0.4 - 0.3 * bias_factor, 0.2, 0.3])
            p /= p.sum()

        length = rng.integers(100, 500) * 3
        seq = "".join(rng.choice(bases, size=length, p=p))

        score = enc_default.get_score(seq)
        all_scores.append(score)

    # We do not store the sequences to keep the regression file small
    # Just the scores which represent the mathematical output
    data = pd.DataFrame({
        "iteration": np.arange(1000),
        "enc_score": all_scores
    })

    # The first time this runs, it creates a baseline file.
    # Subsequent runs compare against that file.
    dataframe_regression.check(data)


# --- PERFORMANCE BENCHMARKS ---
# These are marked so they can be excluded from regular fast test runs

@pytest.mark.benchmark
def test_enc_performance_bottleneck(enc_default, random_seq_gen, capsys):
    """
    Measures the massive speedup of the Cython byte-loop over Pandas.
    Run with: pytest tests/test_scores.py -m benchmark -s
    """
    assert codonbias.scores.HAS_CYTHON is True, "Benchmark requires compiled Cython."

    seq = random_seq_gen(10000)  # 10kb sequence
    iterations = 500

    # Benchmark Pandas
    codonbias.scores.HAS_CYTHON = False
    t0 = time.perf_counter()
    for _ in range(iterations):
        enc_default.get_score(seq)
    time_py = time.perf_counter() - t0

    # Benchmark Cython
    codonbias.scores.HAS_CYTHON = True
    t0 = time.perf_counter()
    for _ in range(iterations):
        enc_default.get_score(seq)
    time_cy = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n[BENCHMARK] {iterations} iterations on 10kb sequence")
        print(f"  Pandas time: {time_py:.4f}s")
        print(f"  Cython time: {time_cy:.4f}s")
        print(f"  Speedup:     {time_py / time_cy:.1f}x")