import warnings

import numpy as np
import pandas as pd

from codonbias import scores, stats


class WeightOptimizer(object):
    """
    Optimizer that uses codon weights to choose between synonymous codons.

    Parameters
    ----------
    strategy : {'max', 'min', 'balanced'}
        Selection strategy:
        - 'max': pick the highest-weight codon per position.
        - 'min': pick the lowest-weight codon per position.
        - 'balanced': sample codons with probability proportional to their
          weight, yielding a balanced distribution where more optimal
          codons appear at higher frequencies.
    weights : pd.Series, optional
        Codon weights, according to which optimization will encode the
        sequence, by default None
    model : scores.ScalarScore, optional
        Codon model object with a `weights` property, by default None
    higher_is_better : bool, optional
        Defines the direction of the weights for the optimization, by
        default True
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    """

    _STRATEGIES = ("max", "min", "balanced")

    def __init__(
        self,
        strategy,
        weights=None,
        model=None,
        higher_is_better=True,
        genetic_code=1,
    ):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._STRATEGIES}, got {strategy!r}"
            )
        self._validate_score(model)

        self.strategy = strategy

        if weights is not None:
            self.weights = weights
        elif model is not None:
            self.weights = model.weights
        else:
            raise TypeError(
                "Optimizer requires either a `weights` argument or a `model`"
            )

        self.higher_is_better = higher_is_better
        self.genetic_code = str(genetic_code)
        self.weights = self._build_synonymous_weights()

    def _validate_score(self, score_object):
        if score_object is None:
            return

        if not isinstance(score_object, scores.ScalarScore):
            raise TypeError(
                f"score_object type is {type(score_object)} and not a ScalarScore"
            )
        if not hasattr(score_object, "weights"):
            raise ValueError("score object does not have a `weights` property")

    def _build_synonymous_weights(self):
        """
        Returns normalized synonymous weights that sum to 1.
        """
        weights = self.weights.rename("weights")

        if not self.higher_is_better:
            weights = 1 / weights

        code = (
            stats.gc[[self.genetic_code]]
            .join(weights)
            .rename(columns={self.genetic_code: "aa"})
            .reset_index()
            .fillna({"weights": 1})
        )
        code = code.merge(
            code.groupby("aa")["weights"].sum().to_frame("norm").reset_index()
        )
        code["weights"] /= code["norm"]

        return code[["aa", "codon", "weights"]]

    def _get_seq_candidates(self, seq_aa):
        """
        Returns all synonymous candidate codons for each position along
        with their optimization weight.

        Parameters
        ----------
        seq_aa : str
            Amino acid sequence.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with the columns `aa, pos, codon, weights`
        """
        seq_aa = list(seq_aa)
        return pd.DataFrame({"aa": seq_aa, "pos": np.arange(len(seq_aa))}).merge(
            self.weights, how="left"
        )

    def optimize(self, seq_aa):
        """
        Encode an amino acid sequence using the configured strategy.

        Parameters
        ----------
        seq_aa : str
            Amino acid sequence.

        Returns
        -------
        str
            DNA sequence.
        """
        candidates = self._get_seq_candidates(seq_aa)
        if self.strategy == "max":
            return "".join(
                candidates.loc[candidates.groupby("pos")["weights"].idxmax(), "codon"]
            )
        if self.strategy == "min":
            return "".join(
                candidates.loc[candidates.groupby("pos")["weights"].idxmin(), "codon"]
            )
        # 'balanced'
        return "".join(
            candidates.groupby("pos").apply(
                lambda df: df.sample(n=1, weights="weights")
            )["codon"]
        )


def _deprecated_optimizer(strategy, old_name):
    warnings.warn(
        f"{old_name} is deprecated; use WeightOptimizer(strategy={strategy!r}) "
        "instead. Will be removed in v0.6.0.",
        FutureWarning,
        stacklevel=3,
    )


class MaxWeight(WeightOptimizer):
    """Deprecated. Use ``WeightOptimizer(strategy='max')``."""

    def __init__(self, **kwargs):
        _deprecated_optimizer("max", "MaxWeight")
        super().__init__(strategy="max", **kwargs)


class MinWeight(WeightOptimizer):
    """Deprecated. Use ``WeightOptimizer(strategy='min')``."""

    def __init__(self, **kwargs):
        _deprecated_optimizer("min", "MinWeight")
        super().__init__(strategy="min", **kwargs)


class BalancedWeight(WeightOptimizer):
    """Deprecated. Use ``WeightOptimizer(strategy='balanced')``."""

    def __init__(self, **kwargs):
        _deprecated_optimizer("balanced", "BalancedWeight")
        super().__init__(strategy="balanced", **kwargs)
