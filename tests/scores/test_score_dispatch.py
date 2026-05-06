"""Contract tests for the shared `Score._dispatch` helper.

The dispatch shell is shared across `ScalarScore.get_score`,
`VectorScore.get_vector` and `WeightScore.get_weights`. Lock its
contract here so future changes don't silently break any of the three.
"""

import numpy as np
import pytest

from codonbias.scores import Score


class _Probe(Score):
    """Minimal Score subclass for exercising _dispatch in isolation."""

    def __init__(self):
        self.calls = []

    def _calc(self, seq, **kwargs):
        self.calls.append((seq, kwargs))
        return len(seq)


def test_dispatch_str_calls_calc_fn():
    p = _Probe()
    assert p._dispatch("ATG", p._calc) == 3
    assert p.calls == [("ATG", {})]


def test_dispatch_str_with_slice():
    p = _Probe()
    assert p._dispatch("ATGCAT", p._calc, slice=slice(3)) == 3
    assert p.calls == [("ATG", {})]


def test_dispatch_list_returns_ndarray_of_results():
    p = _Probe()
    out = p._dispatch(["A", "AT", "ATG"], p._calc)
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_dispatch_ndarray_treated_as_iterable():
    p = _Probe()
    seqs = np.array(["A", "AT", "ATG"])
    out = p._dispatch(seqs, p._calc)
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_dispatch_passes_kwargs_through():
    p = _Probe()
    p._dispatch("ATG", p._calc, foo="bar")
    assert p.calls == [("ATG", {"foo": "bar"})]


def test_dispatch_passes_kwargs_through_iterable():
    p = _Probe()
    p._dispatch(["A", "T"], p._calc, foo="bar")
    assert p.calls == [("A", {"foo": "bar"}), ("T", {"foo": "bar"})]


def test_dispatch_unknown_type_raises():
    p = _Probe()
    with pytest.raises(ValueError, match="unknown sequence type"):
        p._dispatch(42, p._calc)
