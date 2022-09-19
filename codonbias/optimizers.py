import numpy as np
import pandas as pd

from codonbias import scores, stats


class WeightOptimizer(object):
    """
    Abstract class for optimizers that use codon weights to choose
    between synonymous sequences.

    Parameters
    ----------
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
    def __init__(self, weights=None, model=None, higher_is_better=True, genetic_code=1):
        self._validate_score(model)

        if weights is not None:
            self.weights = weights
        elif model is not None:
            self.weights = model.weights
        else:
            raise TypeError('Optimizer requires either a `weights` argument or a `model`')

        self.higher_is_better = higher_is_better
        self.genetic_code = str(genetic_code)
        self.weights = self._build_synonymous_weights()

    def _validate_score(self, score_object):
        if score_object is None:
            return

        if not isinstance(score_object, scores.ScalarScore):
            raise TypeError(f'score_object type is {type(score_object)} and not a ScalarScore')
        if not hasattr(score_object, 'weights'):
            raise ValueError(f'score object does not have a `weights` property')

    def _build_synonymous_weights(self):
        """
        Returns normalized synonymous weights that sum to 1.
        """
        weights = self.weights.rename('weights')

        if not self.higher_is_better:
            weights = 1 / weights

        code = stats.gc[[self.genetic_code]].join(weights)\
            .rename(columns={self.genetic_code: 'aa'})\
            .reset_index().fillna({'weights': 1})
        code = code.merge(code.groupby('aa')['weights']
            .sum().to_frame('norm').reset_index())
        code['weights'] /= code['norm']

        return code[['aa', 'codon', 'weights']]

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
        return pd.DataFrame({'aa': seq_aa, 'pos': np.arange(len(seq_aa))})\
            .merge(self.weights, how='left')

    def optimize(self, seq_aa):
        raise Exception('not implemented')


class MaxWeight(WeightOptimizer):
    """
    Optimizes the amino acid sequence by selecting synonymous codons with
    the highest weights.

    Parameters
    ----------
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
    def optimize(self, seq_aa):
        weights = self._get_seq_candidates(seq_aa)
        return ''.join(weights.loc[weights.groupby('pos')
            ['weights'].idxmax(), 'codon'])


class MinWeight(WeightOptimizer):
    """
    Optimizes the amino acid sequence by selecting synonymous codons with
    the lowest weights.

    Parameters
    ----------
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
    def optimize(self, seq_aa):
        weights = self._get_seq_candidates(seq_aa)
        return ''.join(weights.loc[weights.groupby('pos')
            ['weights'].idxmin(), 'codon'])


class BalancedWeight(WeightOptimizer):
    """
    Optimizes the amino acid sequence by selecting synonymous codons with
    a probability proportional to their weight. This generates a balanced
    codon distribution, with more optimal codons appearing at higher
    frequencies.

    Parameters
    ----------
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
    def optimize(self, seq_aa):
        weights = self._get_seq_candidates(seq_aa)
        return ''.join(weights.groupby('pos')
            .apply(lambda df: df.sample(n=1, weights='weights'))['codon'])
