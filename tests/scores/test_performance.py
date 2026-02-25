import pytest
import time


@pytest.mark.benchmark
def test_enc_performance_bottleneck(enc_default, random_seq_gen, capsys):
    """
    Measures the massive speedup of the Cython byte-loop over Pandas.
    Run with: pytest tests/test_scores.py -m benchmark -s
    """
    seq = random_seq_gen(10000)  # 10kb sequence
    iterations = 500

    # Benchmark Cython
    t0 = time.perf_counter()
    for _ in range(iterations):
        enc_default.get_score(seq)
    time_py = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n[BENCHMARK] {iterations} iterations on 10kb sequence")
        print(f"  Time: {time_py:.4f}s")
