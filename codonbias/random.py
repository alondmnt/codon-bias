import psutil
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from .utils import translate, greater_equal, less_equal, rankdata
from .scores import VectorScore


class Permuter(object):
    """
    This general-prupose permuter generates random sequences by shuffling
    codons within and between sequences while preserving a defined
    property of the sequence. This null model can be used to return the
    shuffled sequences, or to estimate the z-score / p-value of weight
    vectors associated with the sequence.

    The property (or properties) to be preserved by the permutation
    is defined using `property_func`. For example, the default
    `property_func` translates the sequence to amino acids, and therefore
    the permutation preserves the amino acid sequence. However, arbitrary
    properties may be defined. When `n_samples` equals zero, the permuter
    attemps to estimate the z-scores and p-values without actually
    permuting the sequences (very fast). This is especially useful and
    accurate for computing z-scores. While the resulting p-values are
    highly correlated with permutation results, they tend to be
    lower than permutation p-values by 30% on average (but up to 60% lower
    at most).

    Parameters
    ----------
    property_func : function, optional
        Property generating function that accepts a sequence as input and
        returns a pandas.DataFrame with propery columns, by default
        codonbias.utils.translate
    n_samples : int, optional
        The numper of permutations to generate for each sequence. When
        zero, the permuter attempts to estimate the z-scores and
        p-values without actually permuting the sequences, by default 100
    random_state : int, optional
        Random seed for the permutation function, by default 42
    n_jobs : int or None, optional
        Number of parallel processes to run. When set to None the permuter
        will use the number of available cores, by default None
    kwargs :
        Parameters to be passed to the `property_func`.

    See also
    --------
    codonbias.random.IntraSeqPermuter : Within-sequence permutation.
    codonbias.random.IntraPosPermuter : Positional permutation.
    """
    def __init__(self, property_func=translate, add_properties=[],
                 n_samples=100, random_state=42, n_jobs=None, **kwargs):
        self.property_func = property_func
        self.property_args = kwargs
        self.add_properties = add_properties
        self.n_samples = n_samples
        self.random_state = random_state

        if n_jobs is None:
            n_jobs = psutil.cpu_count(logical=False)
        self.n_jobs = n_jobs
        pandarallel.initialize(nb_workers=self.n_jobs, verbose=1)

    def get_permuted_seq(self, seqs, slice=None):
        """
        Computes `n_samples` permutations of the given sequences.

        Parameters
        ----------
        seqs : iterable of str
            DNA sequence.
        slice : slice object, optional
            Optional slicing applied to all sequences prior to
            perpmuation, by deafult None

        Returns
        -------
        pandas.DataFrame
            Permuted sequences DataFrame with `n_samples` columns.
        """
        if slice is not None:
            seqs = [s[slice] for s in seqs]

        df, prop_cols = self._preprocess_df(seqs)

        return self._postprocess_seq(self._permute_df(
            df, prop_cols, 'seq', self.n_samples))

    def get_zscore(self, vector, seqs,
                   slice=None, mapfunc=None, aggfunc=None,
                   model_kws={}):
        """
        Compute the z-score for each position in the vector using random
        permutations of the sequences. The parameter `vector` can be either
        a weights vector or a VectorScore model. If the latter is provided,
        the weights will be recomputed for each permuted sequence (slower),
        otherwise the weights vector itself will be permuted (faster).

        Parameters
        ----------
        vector : iterable or scores.VectorScore
            Weights to be permuted in order to compute the z-score, or a
            VectorScore model.
        seqs : iterable of str
            DNA sequence.
        slice : slice object, optional
            Optional slicing applied to all sequences and vectors prior to
            permutation, by deafult None
        mapfunc : function, optional
            Optional map function to be applied to every vector, by
            default None
        aggfunc : function, optional
            Optional agg function to aggregate all vectors, by default
            None
        model_kws : dict, optional
            Optional keyword arguments to the VectorScore model's
            get_vector function, by default {}

        Returns
        -------
        pandas.Series
            Z-scores series with an entry for each input sequence that
            contains its z-scores array.
        """
        if slice is not None:
            seqs = [s[slice] for s in seqs]

        if isinstance(vector, VectorScore):
            return self._permute_and_compute(vector, seqs, return_pval=False,
                                             mapfunc=mapfunc, aggfunc=aggfunc,
                                             model_kws=model_kws)

        if mapfunc is not None or aggfunc is not None:
            raise TypeError(f'`mapfunc/aggfunc` are only supported when `vector` is a VectorScore model')
        if slice is not None:
            vector = [v[slice] for v in vector]
        return self._permute_vector(vector, seqs, return_pval=False)

    def get_pval(self, vector, seqs, alternative='greater',
                 slice=None, mapfunc=None, aggfunc=None,
                 model_kws={}):
        """
        Compute the p-value for each position in the vector using random
        permutations of the sequences. The parameter `vector` can be either
        a weights vector or a VectorScore model. If the latter is provided,
        the weights will be recomputed for each permuted sequence (slower),
        otherwise the weights vector itself will be permuted (faster).

        Parameters
        ----------
        vector : iterable or scores.VectorScore
            Weights to be permuted in order to compute the z-score, or a
            VectorScore model.
        seqs : iterable of str
            DNA sequence.
        slice : slice object, optional
            Optional slicing applied to all sequences and vectors, by
            deafult None
        mapfunc : function, optional
            Optional map function to be applied to every vector, by
            default None
        aggfunc : function, optional
            Optional agg function to aggregate all vectors, by default
            None
        model_kws : dict, optional
            Optional keyword arguments to the VectorScore model's
            get_vector function, by default {}

        Returns
        -------
        pandas.Series
            Z-scores series with an entry for each input sequence that
            contains its p-values array.
        """
        if slice is not None:
            seqs = [s[slice] for s in seqs]

        if isinstance(vector, VectorScore):
            return self._permute_and_compute(vector, seqs, return_pval=True,
                                             alternative=alternative,
                                             mapfunc=mapfunc, aggfunc=aggfunc,
                                             model_kws=model_kws)

        if mapfunc is not None or aggfunc is not None:
            raise TypeError(f'`mapfunc/aggfunc` are only supported when `vector` is a VectorScore model')
        if slice is not None:
            vector = [v[slice] for v in vector]
        return self._permute_vector(vector, seqs, return_pval=True,
                                    alternative=alternative)

    def _preprocess_df(self, seqs, **kwargs):
        """
        Compute the property for all sequences and return them in
        as a single DataFrame, along with the list of columns containing
        the property to be conserved.
        """
        df = pd.concat([self._preprocess_seq(s, **{k: v[i] for k, v in kwargs.items()})
                        .assign(id=i)
                        for i, s in enumerate(seqs)], axis=0).reset_index(drop=True)
        prop_cols = [col for col in df.columns
                     if col not in ['seq', 'id', 'pos'] + list(kwargs.keys())] \
                    + self.add_properties
        print(f'generating permutations preserving the properties: {prop_cols}')

        return df, prop_cols

    def _preprocess_seq(self, seq, **kwargs):
        """
        Compute the property for a single sequence.
        """
        prop = self.property_func(seq, **self.property_args)  # may have multiple columns
        df = {'seq': [seq[i:i+3] for i in range(0, len(seq), 3)],
              'pos': np.arange(len(seq) // 3)}
        df.update(kwargs)
        return pd.DataFrame(df)\
            .join(prop.reset_index(drop=True))

    def _postprocess_seq(self, df):
        """
        Join each set of permuted codons to a single cohesive sequence
        string.
        """
        cols = [col for col in df.columns if col[:5] == 'null_']
        return df.sort_values('pos').groupby('id').parallel_apply(
            lambda df: df[cols].apply(lambda x: ''.join(x)))

    def _permute_df(self, df, by, col, n):
        """
        Return `n` permutations of a dataframe based on the property
        column(s) `by` and a value column `col`.
        """
        return df.groupby(by).parallel_apply(
            lambda x: self._permute_col(x, col, n))\
            .droplevel(by)

    def _permute_col(self, df, col, n):
        """
        Returns a DataFrame with `n` permutations of the column `col`.
        """
        np.random.seed(self.random_state)
        for i in range(n):
            df[f'null_{i}'] = np.random.permutation(df[col])
        return df

    def _permute_vector(self, vector, seqs,
                        return_pval=False, alternative='greater'):
        """
        This function tends to be efficient, however it assumes that the
        computation of vector values is order invariant. Otherwise, the
        vector has to be computed for each permuted sequence from scratch.

        When `n_samples` is 0, the z-scores / p-values are estimated
        without permuting the vector (very fast). This is especially
        useful and accurate for computing z-scores. While p-values are
        highly correlated with permutation results, they tend to be
        underestimated by up to 60%.

        See also:
        ---------
        self._permute_and_compute
        """
        alt_values = {'greater', 'less'}
        if alternative not in alt_values:
            raise ValueError(f"alternative must be in {alt_values}, got '{alternative}'")

        df, prop_cols = self._preprocess_df(seqs, weights=vector)

        if self.n_samples == 0:
            return self._skip_permutation(
                df, 'weights', prop_cols, return_pval, alternative)

        null = self._permute_df(df, prop_cols, 'weights', self.n_samples)\
            .sort_values(['id', 'pos']).drop(columns=df.columns)
        if return_pval:
            return self._calc_pval(df, null, alternative)

        df['mean'] = null.mean(axis=1)
        df['std'] = null.std(axis=1)
        df['zscore'] = (df['weights'] - df['mean']) / df['std']

        return df.sort_values(['id', 'pos'])\
            .groupby('id')['zscore'].apply(lambda x: x.values)

    def _permute_and_compute(self, model, seqs,
                             return_pval=False, alternative='greater',
                             mapfunc=None, aggfunc=None,
                             model_kws={}):
        """
        This function does not make any assumptions on the computation of
        the weight vector for each sequence, and recomputes it using the
        given VectorScore model. If the result of a computation of the
        vector on a permuted sequence is identical to a corresponding
        permuted vector, then it is generally faster to use the method
        _permute_vector().

        See also:
        ---------
        self._permute_vector
        """
        alt_values = {'greater', 'less'}
        if alternative not in alt_values:
            raise ValueError(f"alternative must be in {alt_values}, got '{alternative}'")

        if self.n_samples == 0:
            raise ValueError("""`n_samples` cannot be 0 when providing a VectorScore model.
                             Increase `n_samples` or provide a weights vector for the
                             sequence instead.""")

        null = self.get_permuted_seq(seqs)\
            .parallel_applymap(lambda x: model.get_vector(x, **model_kws))
        df = pd.DataFrame({'weights': list(model.get_vector(seqs, **model_kws))},
                          index=null.index)
        if mapfunc is not None:
            null = null.parallel_applymap(mapfunc)
            df['weights'] = df['weights'].apply(mapfunc)
        if aggfunc is not None:
            null = null.agg(aggfunc)
            df = df.agg(aggfunc)

        if return_pval:
            return self._calc_pval(df, null, alternative, iterate=True)

        if type(null) is pd.DataFrame:
            df['mean'] = [np.nanmean(np.vstack(row), axis=0)
                          for _, row in null.iterrows()]
            df['std'] = [np.nanstd(np.vstack(row), axis=0)
                         for _, row in null.iterrows()]
        else:
            df['mean'] = np.nanmean(np.array(list(null)), axis=0)
            df['std'] = np.nanstd(np.array(list(null)), axis=0)
        df['zscore'] = (df['weights'] - df['mean']) / df['std']

        return df['zscore']

    def _skip_permutation(self, df, col, by, return_pval, alternative='greater'):
        """
        Estimate results without bootstrapping / permutations by computing
        statistics within each property bin.
        """
        if return_pval:
            out_col = 'pval'
            if alternative == 'greater':
                func = lambda x: 1 - (rankdata(x) + 1) / (len(x) + 1)
            elif alternative == 'less':
                func = lambda x: (rankdata(x) + 1) / (len(x) + 1)
            elif alternative == 'two-sided':
                func = lambda x: 0.5 - np.abs((rankdata(x) + 1) / (len(x) + 1) - 0.5)
        else:
            out_col = 'zscore'
            func = lambda x: (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df[out_col] = df.groupby(by, sort=False)[col].apply(
                lambda x: pd.DataFrame(func(x.to_frame().values),
                                       index=x.index, columns=[col]))

        return df.sort_values(['id', 'pos'])\
            .groupby('id')[out_col].apply(lambda x: x.values)

    def _calc_pval(self, df, null, alternative, iterate=False):
        if alternative == 'greater':
            func = greater_equal
        elif alternative == 'less':
            func = less_equal

        if iterate:
            if type(null) is pd.DataFrame:
                df['pval'] = [np.sum(func(np.vstack(row), real), axis=0)
                              for real, (_, row) in zip(df['weights'], null.iterrows())]
            else:
                df['pval'] = np.sum(func(np.array(list(null)), df['weights']), axis=0)
            df['pval'] = (df['pval'] + 1) / (self.n_samples + 1)
            return df['pval']

        df['pval'] = func(null, df[['weights']].values).sum(axis=1)
        df['pval'] = (df['pval'] + 1) / (self.n_samples + 1)
        df.loc[df['weights'].isnull(), 'pval'] = np.nan

        return df.sort_values(['id', 'pos'])\
            .groupby('id')['pval'].apply(lambda x: x.values)


class IntraSeqPermuter(Permuter):
    """
    This permuter generates random sequences by shuffling the codons
    within each sequence while preserving a defined property of the
    sequence. This null model can be used to return the shuffled
    sequences, or to estimate the z-score / p-value of weight vectors
    associated with the sequence.

    The property (or properties) to be preserved by the permutation
    is defined using `property_func`. For example, the default
    `property_func` translates the sequence to amino acids, and therefore
    the permutation preserves the amino acid sequence. However, arbitrary
    properties may be defined. When `n_samples` equals zero, the permuter
    attemps to estimate the z-scores and p-values without actually
    permuting the sequences (very fast). This is especially useful and
    accurate for computing z-scores. While the resulting p-values are
    highly correlated with permutation results, they tend to be
    lower than permutation p-values by 30% on average (but up to 60% lower
    at most).

    Parameters
    ----------
    property_func : function, optional
        Property generating function that accepts a sequence as input and
        returns a pandas.DataFrame with propery columns, by default
        utils.translate
    n_samples : int, optional
        The numper of permutations to generate for each sequence. When
        zero, the permuter attempts to estimate the z-scores and
        p-values without actually permuting the sequences, by default 100
    random_state : int, optional
        Random seed for the permutation function, by default 42
    n_jobs : int or None, optional
        Number of parallel processes to run. When set to None the permuter
        will use the number of available cores, by default None
    kwargs :
        Parameters to be passed to the `property_func`.

    See also
    --------
    codonbias.random.Permuter : General-purpose permutation.
    codonbias.random.IntraPosPermuter : Positional permutation.
    """
    def __init__(self, property_func=translate,
                 n_samples=100, random_state=42, n_jobs=None, **kwargs):
        super().__init__(property_func=property_func, add_properties=['id'],
                 n_samples=n_samples, random_state=random_state, n_jobs=n_jobs, **kwargs)


class IntraPosPermuter(Permuter):
    """
    This permuter generates random sequences by shuffling codons in each
    position between all sequences while preserving a defined property of
    the sequence. This null model can be used to return the shuffled
    sequences, or to estimate the z-score / p-value of weight vectors
    associated with the sequence.

    The property (or properties) to be preserved by the permutation
    is defined using `property_func`. For example, the default
    `property_func` translates the sequence to amino acids, and therefore
    the permutation preserves the amino acid sequence. However, arbitrary
    properties may be defined. When `n_samples` equals zero, the permuter
    attemps to estimate the z-scores and p-values without actually
    permuting the sequences (very fast). This is especially useful and
    accurate for computing z-scores. While the resulting p-values are
    highly correlated with permutation results, they tend to be
    lower than permutation p-values by 30% on average (but up to 60% lower
    at most).

    Parameters
    ----------
    property_func : fuction, optional
        Property generating function that accepts a sequence as input and
        returns a pandas.DataFrame with propery columns, by default
        utils.translate
    n_samples : int, optional
        The numper of permutations to generate for each sequence. When
        zero, the permuter attempts to estimate the z-scores and
        p-values without actually permuting the sequences, by default 100
    random_state : int, optional
        Random seed for the permutation function, by default 42
    n_jobs : int or None, optional
        Number of parallel processes to run. When set to None the permuter
        will use the number of available cores, by default None
    kwargs :
        Parameters to be passed to the `property_func`.

    See also
    --------
    codonbias.random.Permuter : General-purpose permutation.
    codonbias.random.IntraSeqPermuter : Within-sequence permutation.
    """
    def __init__(self, property_func=translate,
                 n_samples=100, random_state=42, n_jobs=None, **kwargs):
        super().__init__(property_func=property_func, add_properties=['pos'],
                 n_samples=n_samples, random_state=random_state, n_jobs=n_jobs, **kwargs)
