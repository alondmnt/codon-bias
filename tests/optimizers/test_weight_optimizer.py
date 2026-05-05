import warnings

import numpy as np
import pandas as pd
import pytest

from codonbias import stats
from codonbias.optimizers import (
    BalancedWeight,
    MaxWeight,
    MinWeight,
    WeightOptimizer,
)


@pytest.fixture
def codon_weights():
    """Reproducible random codon weights for the standard genetic code."""
    codons = stats.gc[["1"]].reset_index().rename(columns={"1": "aa", "index": "codon"})
    codons = codons[codons["aa"] != "*"]
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.random(len(codons)),
        index=codons["codon"].values,
        name="weights",
    )


@pytest.fixture
def aa_seq():
    return "MKLAFIPVTRGYHN"


@pytest.mark.parametrize("strategy", ["max", "min", "balanced"])
def test_optimize_returns_codon_string_of_correct_length(
    codon_weights, aa_seq, strategy
):
    np.random.seed(0)
    out = WeightOptimizer(strategy=strategy, weights=codon_weights).optimize(aa_seq)
    assert isinstance(out, str)
    assert len(out) == 3 * len(aa_seq)


def test_max_picks_higher_weight_than_min(codon_weights, aa_seq):
    """Per-position, the 'max' codon weight must be >= the 'min' codon weight."""
    opt_max = WeightOptimizer(strategy="max", weights=codon_weights)
    opt_min = WeightOptimizer(strategy="min", weights=codon_weights)
    seq_max = opt_max.optimize(aa_seq)
    seq_min = opt_min.optimize(aa_seq)

    w = opt_max.weights.set_index("codon")["weights"]
    for i in range(len(aa_seq)):
        c_max = seq_max[3 * i : 3 * (i + 1)]
        c_min = seq_min[3 * i : 3 * (i + 1)]
        assert w[c_max] >= w[c_min]


def test_balanced_is_deterministic_under_numpy_seed(codon_weights, aa_seq):
    opt = WeightOptimizer(strategy="balanced", weights=codon_weights)
    np.random.seed(0)
    a = opt.optimize(aa_seq)
    np.random.seed(0)
    b = opt.optimize(aa_seq)
    assert a == b


def test_unknown_strategy_raises(codon_weights):
    with pytest.raises(ValueError, match="strategy must be one of"):
        WeightOptimizer(strategy="best", weights=codon_weights)


def test_missing_weights_and_model_raises():
    with pytest.raises(TypeError, match="weights"):
        WeightOptimizer(strategy="max")


@pytest.mark.parametrize(
    ("shim_cls", "strategy"),
    [
        (MaxWeight, "max"),
        (MinWeight, "min"),
        (BalancedWeight, "balanced"),
    ],
)
def test_shim_emits_future_warning_and_matches_new_api(
    codon_weights, aa_seq, shim_cls, strategy
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.random.seed(0)
        shim_out = shim_cls(weights=codon_weights).optimize(aa_seq)

    assert any(issubclass(w.category, FutureWarning) for w in caught), (
        f"{shim_cls.__name__} did not emit FutureWarning"
    )

    np.random.seed(0)
    new_out = WeightOptimizer(strategy=strategy, weights=codon_weights).optimize(aa_seq)
    assert shim_out == new_out
