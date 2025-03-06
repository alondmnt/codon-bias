from itertools import product
import os

import numpy as np
import pandas as pd
from scipy import stats, optimize

from .stats import CodonCounter, BaseCounter
from .utils import fetch_GCN_from_GtRNAdb, geomean, mean, reverse_complement


class ScalarScore(object):
    """
    Abstract class for models that output a scalar per sequence.
    Inheriting classes may implement the computation of the score for
    a single sequence in the method `_calc_score(seq)`. Parameters
    of the model may be initialized with the instance of the class.
    """
    def __init__(self):
        pass

    def get_score(self, seq, slice=None, **kwargs):
        """
        Compute the score for a single, or multiple sequences. When
        `slice` is provided, all sequences will be sliced before
        computing the score.

        Parameters
        ----------
        seq : str or an iterable of str
            DNA sequence, or an iterable of ones.
        slice : slice, optional
            Python slice object, by default None

        Returns
        -------
        float or numpy.array
            Score for each provided sequence.

        Examples
        --------
        >>> EffectiveNumberOfCodons().get_score('ACGACGGAGGAG')
        35.0

        >>> EffectiveNumberOfCodons().get_score('ACGACGGAGGAG', slice=slice(6))
        44.33333333333333
        """
        if not isinstance(seq, str):
            return np.array([self.get_score(s, slice=slice, **kwargs) for s in seq])

        if slice is not None:
            return self._calc_score(seq[slice], **kwargs)
        else:
            return self._calc_score(seq, **kwargs)

    def _calc_score(self, seq):
        raise Exception('not implemented')


class VectorScore(object):
    """
    Abstract class for models that output a vector per sequence. For
    example, the output can be a score per position in the sequence.
    Inheriting classes may implement the computation of the score for
    a single sequence in the method `_calc_vector(seq)`. Parameters
    of the model may be initialized with the instance of the class.
    """
    def __init__(self):
        pass

    def get_vector(self, seq, slice=None, pad=False, **kwargs):
        """
        Compute the score vector for a single, or multiple sequences.
        When `slice` is provided, all sequences will be sliced before
        computing the score.

        Parameters
        ----------
        seq : str or an iterable of str
            DNA sequence, or an iterable of ones.
        slice : slice, optional
            Python slice object, by default None.
        pad : bool, optional
            Pad the vector with NaNs if the sequence is shorter than
            the maximum length, by default False.

        Returns
        -------
        numpy.array, or numpy.array of numpy.array
            1D array for a single sequence, 1D array of 1D arrays for
            arbitrary sequences, or a matrix NxM for N sequences of length
            M.
        """
        if isinstance(seq, str):
            if slice is not None:
                return self._calc_vector(seq[slice], **kwargs)
            else:
                return self._calc_vector(seq, **kwargs)

        dtype = object if slice is None \
            and np.unique([len(s) for s in seq]).size > 1 and not pad else None
        vecs = [self.get_vector(s, slice=slice, **kwargs) for s in seq]

        if pad:
            max_len = max([len(v) for v in vecs])
            vecs = [np.pad(v, (0, max_len - len(v)),
                           mode='constant', constant_values=np.nan) for v in vecs]

        return np.array(vecs, dtype=dtype)

    def _calc_vector(self, seq):
        raise Exception('not implemented')

    def _get_codon_vector(self, seq, k_mer=1):
        return [seq[i:i+3*k_mer] for i in range(0, len(seq), 3)]


class WeightScore(object):
    """
    Abstract class for models that output a weights vector per sequence.
    Inheriting classes may implement the computation of the score for
    a single sequence in the method `_calc_seq_weights(seq)`. Parameters
    of the model may be initialized with the instance of the class.
    """
    def __init__(self):
        pass

    def get_weights(self, seq, slice=None, **kwargs):
        """
        Compute the codon / amino acid weights for a single, or multiple
        sequences. When `slice` is provided, all sequences will be sliced
        before computing the score.

        Parameters
        ----------
        seq : str or an iterable of str
            DNA sequence, or an iterable of ones.
        slice : slice, optional
            Python slice object, by default None

        Returns
        -------
        numpy.array
            N by C array with a weights vector for each of the N provided
            sequences.
        """
        if not isinstance(seq, str):
            return np.array([self.get_weights(s, slice=slice, **kwargs) for s in seq])

        if slice is not None:
            return self._calc_seq_weights(seq[slice], **kwargs)
        else:
            return self._calc_seq_weights(seq, **kwargs)

    def _calc_seq_weights(self, seq):
        raise Exception('not implemented')


class FrequencyOfOptimalCodons(ScalarScore, VectorScore):
    """
    Frequency of Optimal Codons (FOP, Ikemura, J Mol Biol, 1981).

    This model determines the optimal codons for each amino acid based
    on one of two ways:
    1. Their frequency in the given set of reference sequences
    `ref_seq`. This is an approximate score, as the original study
    determined which codons are optimal based on tRNA abundances.
    2. Using codon weights provided in `weights`. These weights can be,
    for example, tAI weights (that are based on tRNA copy numbers).

    Multiple codons may be selected as optimal based on `thresh`. The
    score for a sequence is the fraction of codons in the sequence deemed
    optimal. The returned vector for a sequence is a binary array where
    optimal positions contain 1 and non-optimal ones contain 0.

    Parameters
    ----------
    ref_seq : iterable of str, optional
        A set of reference DNA sequences for codon usage statistics. If
        provided, codon frequencies in the reference set will be used to
        select the optimal codons.
    weights : pandas.DataFrame or pandas.Series, optional
        A DataFrame / Series with codon weights. If provided, the weights
        will be used to select the optimal codons.
    thresh : float, optional
        Minimal ratio between the frequency of a codon and the most
        frequent one in order to be set as optimal, by default 0.8
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default True
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies. this is
        effective when `ref_seq` contains few short sequences. by default 1
    """
    def __init__(self, ref_seq='', weights=None, thresh=0.8, genetic_code=1,
                 ignore_stop=True, pseudocount=1):
        self.thresh = thresh
        self.counter = CodonCounter(genetic_code=genetic_code,
                                    ignore_stop=ignore_stop)
        self.pseudocount = pseudocount
        self.weights = self.counter.count(ref_seq)\
            .get_aa_table(normed=True, pseudocount=pseudocount)
        if ref_seq is not None and len(ref_seq) > 0:
            pass
        elif weights is not None:
            # Ensure that weights have the same index as the counter
            try:
                if type(weights) == pd.Series:
                    weights = weights.to_frame('weights')
                if 'aa' not in weights.index.names:
                    self.weights = weights.join(
                        self.weights.rename('dummy'))\
                        .drop(columns=['dummy'])
                self.weights = self.weights.iloc[:, 0]
            except KeyError:
                raise ValueError('ensure that weights is properly formatted, with levels [codon] or [aa, codon]')
        else:
            raise ValueError('either ref_seq or weights must be provided')

        self.weights = self.weights\
                .groupby('aa', group_keys=False)\
                .apply(lambda x: x / x.max())
        self.weights[self.weights >= self.thresh] = 1  # optimal
        self.weights[self.weights < self.thresh] = 0  # non-optimal
        self.weights = self.weights.droplevel('aa')

    def _calc_score(self, seq):
        counts = self.counter.count(seq).counts

        return mean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.reindex(self._get_codon_vector(seq)).values


class RelativeSynonymousCodonUsage(ScalarScore, VectorScore, WeightScore):
    """
    Relative Synonymous Codon Usage (RSCU, Sharp & Li, NAR, 1986).

    This model measures the deviation of synonymous codon usage from
    uniformity and returns for each codon the ratio between its
    observed frequency and its expected frequency if synonymous codons
    were chosen randomly (uniformly). Overepresented codons will have
    a score > 1, while underrepresented codons will have a score < 1.
    `get_weights()` returns a vector of 61 RSCU ratios for each sequence.
    While not defined as part of the original Sharp & Li model, the
    `get_vector()` method returns an array with the ratio of the
    corresponding codon in each position in the sequence, and the
    `get_score()` method returns the geometric mean of the ratios for a
    sequence (minus 1), in a similar way to the Relative Codon Bias Score
    (RCBS). The `directional` parameter modifies RSCU similarly to the way
    the Directional Codon Bias Score (DCBS) modifies RCBS, by giving
    higher weights to both overrepresented and underrepresented codons.

    Parameters
    ----------
    ref_seq : iterable of str, optional
        When given, codon frequencies in the reference set
        will be used instead of the uniform codon distribution,
        by default None
    directional : bool, optional
        When True will compute the modified version by Sabi & Tuller, by
        default False
    mean : {'geometric', 'arithmetic'}, optional
        How to compute the score, by default 'geometric'
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default True
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies, by
        default 1

    See Also
    --------
    codonbias.scores.RelativeCodonBiasScore

    codonbias.scores.EffectiveNumberOfCodons
    """
    def __init__(self, ref_seq=None, directional=False, mean='geometric',
                 genetic_code=1, ignore_stop=True, pseudocount=1):
        self.directional = directional
        self.mean = mean
        self.counter = CodonCounter(sum_seqs=False, genetic_code=genetic_code,
                                    ignore_stop=ignore_stop)
        self.pseudocount = pseudocount

        if ref_seq is None:
            self.reference = CodonCounter('', genetic_code=genetic_code,
                                          ignore_stop=ignore_stop)
        else:
            self.reference = CodonCounter(ref_seq, genetic_code=genetic_code,
                                          ignore_stop=ignore_stop)
        self.reference = self.reference.get_aa_table(
            normed=True, pseudocount=pseudocount)

    def _calc_score(self, seq):
        D = self._calc_seq_weights(seq).droplevel('aa')
        counts = self.counter.counts  # counts have already been prepared in _calc_weights

        if self.mean == 'geometric':
            return geomean(np.log(D), counts) - 1
        elif self.mean == 'arithmetic':
            return mean(D, counts)
        else:
            raise ValueError(f'unknown mean: {self.mean}')

    def _calc_vector(self, seq):
        weights = self._calc_seq_weights(seq).droplevel('aa')
        return weights.reindex(self._get_codon_vector(seq)).values

    def _calc_seq_weights(self, seq):
        P = self.counter.count(seq)\
            .get_aa_table(normed=True, pseudocount=self.pseudocount)
        # codon weights
        if self.directional:
            D = np.maximum(
                P.divide(self.reference, axis=0),
                self.reference.divide(P, axis=0))
        else:
            D = P.divide(self.reference, axis=0)

        return D


class CodonAdaptationIndex(ScalarScore, VectorScore):
    """
    Codon Adaptation Index (CAI, Sharp & Li, NAR, 1987).

    This model determines the level of optimality of codons based on
    their frequency in the given set of reference sequences `ref_seq`.
    For each amino acid, the most frequent synonymous codon receives
    a weight of 1, while other codons are weighted based on their
    relative frequency with respect to the most frequent synonymous
    codon. The returned vector for a sequence is an array with the
    weight of the corresponding codon in each position in the
    sequence. The score for a sequence is the geometric mean of these
    weights, and ranges from 0 (strong rare codon bias) to 1 (strong
    frequent codon bias).

    This implementation extends the model to arbitrary codon k-mers
    using the `k_mer` parameter.

    Parameters
    ----------
    ref_seq : iterable of str
        Reference sequences for learning the codon frequencies.
    k_mer : int, optional
        Determines the length of the k-mer to base statistics on, by
        default 1
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default True
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies. this is
        effective when `ref_seq` contains few short sequences. by default 1
    """
    def __init__(self, ref_seq, k_mer=1, genetic_code=1,
                 ignore_stop=True, pseudocount=1):
        self.counter = CodonCounter(k_mer=k_mer, concat_index=True,
                                    genetic_code=genetic_code,
                                    ignore_stop=ignore_stop)
        self.k_mer = k_mer
        self.pseudocount = pseudocount

        self._calc_weights(ref_seq)

    def _calc_score(self, seq):
        counts = self.counter.count(seq).counts

        return geomean(self.log_weights, counts)

    def _calc_vector(self, seq):
        return self.weights.reindex(
            self._get_codon_vector(seq, k_mer=self.k_mer)).values

    def _calc_weights(self, seqs):
        self.weights = self.counter.count(seqs)\
            .get_aa_table(normed=True, pseudocount=self.pseudocount)

        aa_levels = [n for n in self.weights.index.names if 'aa' in n]
        self.weights = self.weights.groupby(aa_levels, group_keys=False)\
            .apply(lambda x: x / x.max()).droplevel(aa_levels)

        self.log_weights = np.log(self.weights)


class EffectiveNumberOfCodons(ScalarScore, WeightScore):
    """
    Effective Number of Codons (ENC, Wright, Gene, 1990).

    This model measures the deviation of synonymous codon usage from
    uniformity based on a statistical model analogous to the effective
    number of alleles in genetics. The score for a sequence is the
    effective number of codons in use, and ranges from 20 (very strong
    bias: a single codon per amino acid) to 61 (uniform use of all
    codons). Thus, this score is expected to be negatively correlated
    with most other codon bias measures.

    The model has also been extended to codon pairs by Alexaki et al.
    (JMB, 2019). The `k_mer` parameter can be used to calculate ENC for
    codon pairs as well as longer k-mers.

    When `bg_correction` is True, a background correction procedure is
    performed as proposed by Novembre (MBE, 2002). This procedure
    estimates the background codon composition of each sequence using
    the independent probabilities of observing each of the 4 bases in
    the 3 codon positions. This implementation learns the nucleotide
    probabilities from the provided coding sequence. However, if the
    parameter `background` is given to get_score(), this background
    sequence will be used instead.

    The parameters `robust`, `pseudocount` and `mean` introduce additional
    improvements to the estimation of the effective number as proposed by
    Sun, Yang & Xia (MBE, 2013). They are activated by default, and
    remove, for example, the strong dependency between ENC and sequence
    length.

    Parameters
    ----------
    k_mer : int, optional
        Extends the model to codon k-mers. For example, codon pairs, as
        suggested by Alexaki et al. (JMB, 2019), by default 1
    bg_correction : bool, optional
        Background correction based on Novembre (MBE, 2002), by default
        False
    robust : bool, optional
        Robust estimation of F values that is less sensitive to small
        counts. Proposed improvement by Sun, Yang & Xia (MBE, 2013), by
        default True
    pseudocount : int, optional
        Pseudocounts added to codon statistics. Proposed improvement by
        Sun, Yang & Xia (MBE, 2013), by default 1
    mean : {'weighetd', 'unweighted'}, optional
        Weighted average of F across amino acids by their frequency.
        Proposed improvement by Sun, Yang & Xia (MBE, 2013), by default
        'weighetd'
    genetic_code : int, optional
        NCBI genetic code ID, by default 1

    See Also
    --------
    codonbias.scores.RelativeSynonymousCodonUsage

    codonbias.scores.RelativeCodonBiasScore
    """
    def __init__(self, k_mer=1, bg_correction=False, robust=True,
                 pseudocount=1, mean='weighted', genetic_code=1):
        self.k_mer = k_mer
        self.bg_correction = bg_correction
        self.robust = robust
        self.pseudocount = pseudocount
        self.mean = mean
        self.counter = CodonCounter(
            k_mer=k_mer, concat_index=True,
            genetic_code=genetic_code, ignore_stop=True)  # score is not defined for STOP codons

        self.template = self.counter.count('').get_aa_table().to_frame()
        self.aa_deg = self.template.groupby('aa').size()

        self.BCC_unif = self._calc_BCC(self._calc_BNC(''))

    def _calc_seq_weights(self, seq, background=None):
        return 1 / self._calc_F(seq, background=background)[0]['F']

    def _calc_score(self, seq, background=None):
        F, P, N = self._calc_F(seq, background=background)

        F['deg'] = self.aa_deg
        F['N'] = N
        deg_count = F.groupby('deg').size().to_frame('deg_count')

        if self.mean == 'unweighted':
            # at least 2 samples from AA to be included
            F = F.loc[(N > 1) & (F['F'] > 1e-6) & np.isfinite(F['F'])]\
                .groupby('deg', group_keys=False).mean().join(deg_count, how='right')
        elif self.mean == 'weighted':
            # weighted mean: Sun, Yang & Xia 2013
            F['F'] = F['F'] * F['N']
            F = F.groupby('deg')['F'].sum() / F.groupby('deg')['N'].sum()
            F = F.to_frame('F').join(deg_count, how='right')
        else:
            raise ValueError(f'unknown mean="{self.mean}"')

        # missing AA cases
        miss_3 = np.isnan(F.loc[3, 'F'])
        F['F'] = F['F'].fillna(1/F.index.to_series())  # use 1/deg
        if miss_3:
            F.loc[3, 'F'] = 0.5*(F.loc[2, 'F'] + F.loc[4, 'F'])

        ENC = (F['deg_count'] / F['F']).sum()
        return min([len(P), ENC]) ** (1/self.k_mer)

    def _calc_F(self, seq, background=None):
        counts = self.counter.count(seq).get_aa_table()
        counts += self.pseudocount  # Sun, Yang & Xia 2013

        N = counts.groupby('aa').sum()
        P = counts / N

        if background is None:
            background = seq
        if self.bg_correction:
            BCC = self._calc_BCC(self._calc_BNC(background))
        else:
            BCC = self.BCC_unif

        if not self.robust:
            chi2 = N * ((P - BCC)**2 / BCC).groupby('aa').sum()  # Novembre 2002
            F = ((chi2 + N - self.aa_deg) / (N - 1) / self.aa_deg).to_frame('F')
            # converges to Wright 1990 for BCC_unif
        else:
            chi2 = ((P - BCC)**2 / BCC).groupby('aa').sum()  # modified Novembre 2002
            F = ((chi2 + 1) / self.aa_deg).to_frame('F')
            # converges to Sun, Yang & Xia 2013 for BCC_unif, i.e., sum(p**2)

        return F, P, N

    def _calc_BNC(self, seq):
        """ Compute the background NUCLEOTIDE composition of the sequence. """
        BNC = BaseCounter(seq).get_table(normed=True)

        return BNC

    def _calc_BCC(self, BNC):
        """ Compute the background CODON composition of the sequence. """
        BCC = self.template.copy()
        BCC['bcc'] = [np.prod([BNC[c] for c in cod])
                      for cod in BCC.index.get_level_values('codon')]
        BCC = BCC['bcc']
        BCC /= BCC.groupby('aa').sum()

        return BCC 


class TrnaAdaptationIndex(ScalarScore, VectorScore):
    """
    tRNA Adaptation Index (tAI, dos Reis, Savva & Wernisch, NAR, 2004).

    This model measures translational efficiency based on the
    availablity of tRNAs (approximated by the gene copy number of each
    tRNA species), and the efficiency of coupling between tRNAs and
    codons (modeled via the set of `s_values` coefficients). Each codon
    receives a weight in [0, 1] that describes its translational
    efficiency. The returned vector for a sequence is an array with
    the weight of the corresponding codon in each position in the
    sequence. The score for a sequence is the geometric mean of these
    weights, and ranges from 0 (low efficiency) to 1 (high efficiency).

    Gene copy numbers can be provided explicitly, or automatically
    downloaded from GtRNAdb.

    The model was originally trained in S. cerevisiae and E. coli
    in order to maximize the correlation with mRNA levels measured via
    microarrays. The model was later refitted using protein abundance
    levels (Tuller et al., Genome Biology, 2011). The `s_values`
    parameter can be used to switch between these coefficients sets or
    provide custom values. Additionally, s_values can be optimized such
    that the correlation of tAI with expression (or a CUB measure) is maximized.
    When analyzing an organism that is a prokaryote, the `prokaryote`
    parameter should be set to True.

    Parameters
    ----------
    tGCN : pandas.DataFrame, optional
        tRNA Gene Copy Numbers given as a DataFrame with the columns
        `anti_codon`, `GCN`, by default None
    url : str, optional
        URL of the relevant page on GtRNAdb, by default None
    genome_id : str, optional
        Genome ID of the organism, by default None
    domain : str, optional
        Taxonomic domain of the organism, by default None
    prokaryote : bool, optional
        Whether the organism is a prokaryote, by default False
    s_values : {'dosReis', 'Tuller'} or DataFrame, optional
        Coefficients of the tRNA-codon efficiency of coupling, by default 'dosReis'
        If {'dosReis', 'Tuller'}, default values optimized in yeast are used.
        If DataFrame, s_values are used as provided. Required columns: `anti`, `cod`, `min_deg`, `weight`
    genetic_code : int, optional
        NCBI genetic code ID, by default 1

    See Also
    --------
    codonbias.scores.NormalizedTranslationalEfficiency
    """
    def __init__(self, tGCN=None, url=None, genome_id=None, domain=None,
                 prokaryote=False, s_values='dosReis', genetic_code=1):
        self.counter = CodonCounter(genetic_code=genetic_code,
                                    ignore_stop=True)  # score is not defined for STOP codons

        # tRNA gene copy numbers of the organism
        if url is not None or (genome_id is not None and domain is not None):
            tGCN = fetch_GCN_from_GtRNAdb(url=url, domain=domain, genome=genome_id)
        if tGCN is None:
            raise TypeError('must provide either: tGCN dataframe, GtRNAdb url, or GtRNAdb genome_id+domain')
        tGCN['anti_codon'] = tGCN['anti_codon'].str.upper().str.replace('U', 'T')
        self.tGCN = tGCN

        # S-values: tRNA-codon efficiency of coupling
        if type(s_values) is pd.DataFrame:
            self.s_values = s_values
        elif type(s_values) is str:
            self.s_values = pd.read_csv(
                f'{os.path.dirname(__file__)}/tAI_svalues_{s_values}.csv',
                dtype={'weight': float, 'prokaryote': bool}, comment='#')
        else:
            raise TypeError(f"s_values must be provided as string or dataframe")
        self.s_values['anti'] = self.s_values['anti'].str.upper().str.replace('U', 'T')
        self.s_values['cod'] = self.s_values['cod'].str.upper().str.replace('U', 'T')
        if not prokaryote:
            self.s_values = self.s_values.loc[~self.s_values['prokaryote']]

        self.weights = self._calc_weights()
        self.log_weights = np.log(self.weights)

    def optimize_s_values(self, ref_seq, expression, optimize_wc=False, method="Powell", **kwargs):
        """
        Optimizes s-values such that the Spearman correlation between tAI calculated on ref_seq and expression is maximal.

        Parameters
        ----------
        ref_seq: iterable of str
             Reference sequences for which the correlation between tAI and scores will be maximized.
        expression: iterable of float
            Expression of the provided reference sequences. Can be experimental measurements (original version)
            or CUB measures that are positively correlated with expression (see Sabi & Tuller, DNA Research, 2014)
        optimize_wc: bool
            Whether to optimize s-values for Watson-Crick base pairs, by default False
        method: str
            Optimization algorithm to use, by default 'Powell'
        kwargs:
            Additional parameters to be passed to scipy.optimize.minimize

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result
        """
        if len(ref_seq) != len(expression):
            raise ValueError(
                f'lengths of ref_seq, expression do not match: {len(ref_seq)} != {len(expression)}')
        # Ensure values are finite
        valid = np.isfinite(expression)
        expression = [e for e, v in zip(expression, valid) if v]
        ref_seq = [s for s, v in zip(ref_seq, valid) if v]
        print(f'optimize_s_values: removed {(~valid).sum():,d} non-finite values')

        def func(weights):
            self.s_values["weight"] = weights
            self.weights = self._calc_weights()
            self.log_weights = np.log(self.weights)
            return -stats.spearmanr(self.get_score(ref_seq), expression).statistic

        x0 = np.array(self.s_values["weight"])

        def get_bounds(row):
            if row["anti"] == reverse_complement(row["cod"]) and not optimize_wc:
                return 0, 0
            return 0, 1

        bounds = self.s_values.apply(get_bounds, axis=1).to_list()
        return optimize.minimize(func, x0=x0, bounds=bounds, method=method, **kwargs)

    def _calc_weights(self):
        # init the dataframe
        weights = self.counter.count('').get_aa_table().to_frame('count')
        weights = weights.join(weights.groupby('aa', group_keys=False).size().to_frame('deg'))\
            .reset_index().drop(columns=['aa'])[['codon', 'deg']]
        # columns: codon, deg

        # match all possible tRNAs to codons by the 1st,2nd positions
        weights['cod_12'] = weights['codon'].str[:2]
        self.tGCN['cod_12'] = self.tGCN['anti_codon'].apply(reverse_complement).str[:2]
        weights = weights.merge(self.tGCN, on='cod_12')
        # columns: codon, deg, cod_12, anti_codon, GCN

        # match all possible pairs to S-values by the 3rd position
        weights['anti'] = weights['anti_codon'].str[0]
        weights['cod'] = weights['codon'].str[-1]
        weights = weights.merge(self.s_values, on=['anti', 'cod'])
        weights = weights.loc[weights['deg'] >= weights['min_deg']]
        # columns: codon, deg, cod_12, anti_codon, GCN,
        #          anti, cod, min_deg, weight, prokaryote

        weights['weight'] = (1 - weights['weight']) * weights['GCN']
        weights = weights.groupby('codon')['weight'].sum()

        weights /= weights.max()
        weights[weights == 0] = stats.gmean(
            weights[(weights != 0) & np.isfinite(weights)])

        return weights

    def _calc_score(self, seq):
        counts = self.counter.count(seq).counts

        return geomean(self.log_weights, counts)

    def _calc_vector(self, seq):
        return self.weights.reindex(self._get_codon_vector(seq)).values


class CodonPairBias(ScalarScore, VectorScore, WeightScore):
    """
    Codon Pair Bias (CPB/CPS, Coleman et al., Science, 2008).

    This model is extended here to arbitrary codon k-mers. The model
    calculates the over-/under- represention of codon k-mers compared
    to a background distribution. Each k-mer receives a weight that is the
    log-ratio between its observed and expected probabilities. The
    returned vector for a sequence is an array with the weight of the
    corresponding k-mer in each position in the sequence. The score for a
    sequence is the mean of these weights, and ranges from a negative
    value (mostly under-represented pairs) to a positive value (mostly
    over-represented pairs).

    Parameters
    ----------
    ref_seq : iterable of str
        Reference sequences for learning the codon frequencies.
    k_mer : int, optional
        Determines the length of the k-mer to base statistics on, by
        default 2
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default True
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies. this is
        effective when `ref_seq` contains few short sequences. by default 1

    See Also
    --------
    codonbias.scores.EffectiveNumberOfCodons

    codonbias.scores.CodonAdaptationIndex

    codonbias.pairwise.CodonUsageFrequency
    """
    def __init__(self, ref_seq, k_mer=2, genetic_code=1,
                 ignore_stop=True, pseudocount=1):
        self.counter = CodonCounter(k_mer=k_mer, concat_index=True,
            genetic_code=genetic_code, ignore_stop=ignore_stop)
        self.k_mer = k_mer
        self.pseudocount = pseudocount

        self.weights = self._calc_model_weights(ref_seq)

    def _calc_score(self, seq):
        counts = self.counter.count(seq).counts

        return mean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.reindex(
            self._get_codon_vector(seq, k_mer=self.k_mer)).values

    def _calc_seq_weights(self, seq):
        return self._calc_model_weights(seq)

    def _calc_model_weights(self, seq):
        """
        Calculates the Codon Pair Score (CPS) for each pair (or k-mer).
        That is, the log-ratios of observed over expected frequencies.
        """
        weights = CodonCounter(seq,
            k_mer=self.k_mer, concat_index=False,
            genetic_code=self.counter.genetic_code,
            ignore_stop=self.counter.ignore_stop)\
            .get_aa_table().to_frame('count')
        aa_levels = [n for n in weights.index.names if 'aa' in n]
        cod_levels = [n for n in weights.index.names if 'codon' in n]

        weights['count'] += self.pseudocount
        weights = self._calc_freq(weights, 'aa')
        weights = self._calc_freq(weights, 'codon')

        weights = self._calc_enrichment(weights)\
            .droplevel(aa_levels).reorder_levels(cod_levels)\
            ['log_ratio']
        weights.index = weights.index.to_series()\
            .str.join('')

        return weights

    def _calc_freq(self, counts, word='aa'):
        levels = [n for n in counts.index.names if word in n]

        # calculate k-mer frequencies
        freq_kmer = counts.groupby(levels)['count'].sum()
        freq_kmer /= freq_kmer.sum()
        counts = counts.join(freq_kmer.to_frame(f'freq_{word}_mer'))

        # calculate global frequencies of each "word"
        glob_count = []
        for l in levels:
            glob_count.append(counts.groupby(l)['count'].sum())
        glob_count = pd.concat(glob_count, axis=1).sum(axis=1)
        glob_count /= glob_count.sum()

        # join with the counts dataframe
        for l in levels:
            counts = counts.join(
                glob_count.rename_axis(index=l).to_frame(l))

        # calculate independent joint probabilities
        counts = counts.join(counts[levels].prod(axis=1)
                             .to_frame(f'freq_{word}_ind'))

        return counts

    def _calc_enrichment(self, freqs):
        """
        Calculates the log-ratio between observed k-mer frequencies and
        expected frequencies under the assumption that k-mers are
        distributed independently.
        """
        freqs['log_ratio'] = \
            np.log(freqs['freq_codon_mer'] / freqs['freq_codon_ind']
                   * freqs['freq_aa_ind'] / freqs['freq_aa_mer'])
        return freqs


class RelativeCodonBiasScore(ScalarScore, VectorScore, WeightScore):
    """
    Relative Codon Bias Score (RCBS, Roymondal, Das & Sahoo, DNA Research, 2009).

    This model measures the deviation of codon usage from a background
    distribution and computes for each codon the observed-to-expected
    ratio. The background distribution is estimated for each sequence
    separately, based on its nucleotide composition. The model's null
    hypothesis is that the 3 codon positions are independently
    distributed according to the same nucleotide distribution. Thus,
    overrepresented codons are given higher weights while
    underrepresented codons are given lower weights. The score for a
    sequence is the geometric mean of codon ratios, minus 1. The
    returned vector for a sequence is an array with the ratio of the
    corresponding codon in each position in the sequence.

    Sabi & Tuller (DNA Research, 2014) proposed a modified score based
    on these principles, termed the Directional Codon Bias Score (DCBS).
    In this model underrepresented codons are given larger weights
    (rather than smaller weights) similarly to overrepresnted codons.
    This model's hypothesis is that biased sequences will typically
    include both highly overrepresnted codons as well as
    underrepresented ones, and therefore both signals should
    contribute towards a higher (i.e., biased) score. This
    modification is activated by setting the `directional` parameter
    to True and the `mean` parameter to 'arithmetic'.

    Parameters
    ----------
    directional : bool, optional
        When True will compute the modified version by Sabi & Tuller, by
        default False
    mean : {'geometric', 'arithmetic'}, optional
        How to compute the score, by default 'geometric'
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by
        default True
    pseudocount : int, optional
        Pseudocount correction for normalized codon frequencies, by
        default 1

    See Also
    --------
    codonbias.scores.RelativeSynonymousCodonUsage

    codonbias.scores.EffectiveNumberOfCodons
    """
    def __init__(self, directional=False, mean='geometric',
                 genetic_code=1, ignore_stop=True, pseudocount=1):
        self.directional = directional
        self.mean = mean
        self.pseudocount = pseudocount
        self.counter = CodonCounter(genetic_code=genetic_code,
                                    ignore_stop=ignore_stop)

    def _calc_score(self, seq):
        D = self._calc_seq_weights(seq)
        counts = self.counter.counts  # counts have already been prepared in _calc_seq_weights

        if self.mean == 'geometric':
            return geomean(np.log(D), counts) - 1
        elif self.mean == 'arithmetic':
            return mean(D, counts)
        else:
            raise ValueError(f'unknown mean: {self.mean}')

    def _calc_vector(self, seq):
        D = self._calc_seq_weights(seq)

        return D.reindex(self._get_codon_vector(seq)).values

    def _calc_seq_weights(self, seq):
        counts = self.counter.count(seq)
        # background probabilities
        BCC = self._calc_BCC(self._calc_BNC(seq))
        # observed probabilities
        P = counts.get_codon_table(normed=True, pseudocount=self.pseudocount)
        # codon weights
        if self.directional:
            D = np.maximum(P / BCC, BCC / P)
        else:
            D = P / BCC

        return D

    def _calc_BNC(self, seq):
        """ Compute the background NUCLEOTIDE composition of the sequence. """
        BNC = BaseCounter([seq[i::3] for i in range(3)],
                                sum_seqs=False).get_table()

        return BNC

    def _calc_BCC(self, BNC):
        """ Compute the background CODON composition of the sequence. """
        BCC = pd.DataFrame(
            [(c1+c2+c3, BNC[0][c1] * BNC[1][c2] * BNC[2][c3])
             for c1, c2, c3 in product('ACGT', 'ACGT', 'ACGT')],
            columns=['codon', 'bcc'])
        BCC = BCC.set_index('codon')['bcc']
        BCC /= BCC.sum()

        return BCC


class NormalizedTranslationalEfficiency(ScalarScore, VectorScore):
    """
    Normalized Translational Efficiency (Pechmann & Frydman, Nat. Struct.
    Mol. Biol., 2013)

    This models computes a translational efficiency score that takes into
    account both supply (of tRNAs) and demand (codons being translated).
    Supply is computed based on the tRNA Adaptation Index (tAI), and
    demand is computed based on the sum of all codons in the genome
    weighted by their mRNA abundance (or ribosome occupancy, where
    available). Each codon receives a weight in [0, 1] that describes its
    translational efficiency. The returned vector for a sequence is an
    array with the weight of the corresponding codon in each position in
    the sequence. The score for a sequence is the geometric mean of these
    weights, and ranges from 0 (low efficiency) to 1 (high efficiency).

    Parameters
    ----------
    ref_seq : iterable os str
        Demand parameter: Will be used to count the codons across
        transcripts in a weighted sum
    mRNA_counts : iterable of float
        Demand parameter: Will be used in the weighted sum of codons
        across transcripts
    tGCN : pandas.DataFrame, optional
        Supply parameter: tRNA Gene Copy Numbers given as a DataFrame with
        the columns `anti_codon`, `GCN`, by default None
    url : str, optional
        Supply parameter: URL of the relevant page on GtRNAdb, by default
        None
    genome_id : str, optional
        Supply parameter: Genome ID of the organism, by default None
    domain : str, optional
        Supply parameter: Taxonomic domain of the organism, by default None
    prokaryote : bool, optional
        Supply parameter: Whether the organism is a prokaryote, by default
        False
    s_values : {'dosReis', 'Tuller'}, optional
        Supply parameter: Coefficients of the tRNA-codon efficiency of
        coupling, by default 'dosReis'
    genetic_code : int, optional
        NCBI genetic code ID, by default 1

    See Also
    --------
    codonbias.scores.TrnaAdaptationIndex
    """
    def __init__(self, ref_seq, mRNA_counts, tGCN=None, url=None, genome_id=None,
                 domain=None, prokaryote=False, s_values='dosReis',
                 genetic_code=1):
        if len(ref_seq) != len(mRNA_counts):
            raise ValueError(
                f'lengths of ref_seq, mRNA_counts do not match: {len(ref_seq)} != {len(mRNA_counts)}')

        # supply: classical translational efficiency
        self.tAI = TrnaAdaptationIndex(
            tGCN=tGCN, url=url, genome_id=genome_id, domain=domain,
            prokaryote=prokaryote, s_values=s_values,
            genetic_code=genetic_code)

        # demand: codon usage
        self.CU = CodonCounter(
            ref_seq, sum_seqs=False,
            genetic_code=genetic_code, ignore_stop=True)\
            .get_codon_table()
        self.CU = (self.CU * np.array(mRNA_counts)
                   .reshape(1, -1)).sum(axis=1)  # sum weighted by mRNA counts
        self.CU = self.CU / self.CU.max()

        self.weights = self.tAI.weights / self.CU
        self.weights = self.weights / self.weights.max()
        self.log_weights = np.log(self.weights)

        self.counter = CodonCounter(genetic_code=genetic_code,
                                    ignore_stop=True)

    def _calc_score(self, seq):
        counts = self.counter.count(seq).counts

        return geomean(self.log_weights, counts)

    def _calc_vector(self, seq):
        return self.weights.reindex(self._get_codon_vector(seq)).values
