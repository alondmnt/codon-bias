from collections import Counter
import os

import pandas as pd

gc = pd.read_csv(f'{os.path.dirname(__file__)}/genetic_code_ncbi.csv',
                 index_col=0).sort_index()
# https://en.wikipedia.org/wiki/List_of_genetic_codes


class CodonCounter(object):
    """
    Codon statistics for a single, or multiple DNA sequences.
    When the `k_mer` argument is provided, the counter will return
    codon pairs (k_mer=2), codon triplets (k_mer=3) statistics, etc.

    Parameters
    ----------
    seqs : str, or iterable of str
        DNA sequence, or an iterable of ones.
    k_mer : int, optional
        Determines the length of the k-mer to base statistics on, by
        default 1
    sum_seqs : bool, optional
        Determines how multiple sequences will be handled. When True,
        their statistics will be summed, otherwise separate statistics
        will be kept in a table. by default True
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by default True
    """
    def __init__(self, seqs, k_mer=1, sum_seqs=True, genetic_code=1, ignore_stop=True):
        self.k_mer = k_mer
        self.sum_seqs = sum_seqs
        self.genetic_code = str(genetic_code)
        self.ignore_stop = ignore_stop
        self.counts = self._format_counts(self._count(seqs))
        if self.counts.ndim == 1:
            self.counts = self.counts.rename('count')

    def _count(self, seqs):
        if isinstance(seqs, str):
            return self._count_single(seqs)

        counts = pd.concat([self._count_single(s) for s in seqs], axis=1)
        if self.sum_seqs:
            return counts.sum(axis=1)
        else:
            return counts

    def _count_single(self, seq):
        seq = seq.upper().replace('U', 'T')

        return pd.Series(Counter(
            [seq[i:i + 3*self.k_mer]
             for i in range(0, len(seq), 3)]))

    def _format_counts(self, counts):
        if self.k_mer == 1:
            return counts

        counts.index = pd.MultiIndex.from_arrays([
            counts.index.str[3*k:3*(k+1)]
            for k in range(self.k_mer)],
            names=[f'codon{k}' for k in range(self.k_mer)])

        return counts

    def get_codon_table(self, normed=False, fillna=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False).

        Parameters
        ----------
        normed : bool, optional
            Determines whether codon counts will be normalized to sum to
            1, by default False
        fillna : bool, optional
            When True will fill NaNs according to a unifrom distribution,
            by default False

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Codon counts (or frequencies) with codons as index, and counts
            as values.
        """
        # join genetic code tables for each codon in k-mer
        stats = self.counts
        for k in range(self.k_mer):
            stats = gc[[self.genetic_code]].rename_axis(index=f'codon{k}')\
                .join(stats).fillna(0.)
            if self.ignore_stop:
                stats = stats.loc[stats[self.genetic_code] != '*']
            stats = stats.drop(columns=self.genetic_code)

        if normed:
            stats = stats / stats.sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]
        if fillna:
            stats = stats.fillna(1/len(stats))

        if self.k_mer == 1:
            stats = stats.rename_axis(index='codon')

        return stats

    def get_aa_table(self, normed=False, fillna=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False),
        indexed by the codon and the encoded amino acid.

        Parameters
        ----------
        normed : bool, optional
            Determines whether codon counts will be normalized to sum to
            1 for each amino acid (a vector that sums to 20), by default
            False
        fillna : bool, optional
            When True will fill NaNs according to a unifrom distribution,
            by default False

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Codon counts (or frequencies) with amino acids and codons as
            index, and counts as values.
        """
        # join genetic code tables for each codon in k-mer
        stats = self.counts
        aa_levels = []
        cod_levels = []
        for k in range(self.k_mer):
            aa_levels.append(f'aa{k}')
            cod_levels.append(f'codon{k}')
            stats = gc[[self.genetic_code]]\
                .rename(columns={self.genetic_code: f'aa{k}'})\
                .rename_axis(index=f'codon{k}')\
                .join(stats).fillna(0.)\
                .set_index(f'aa{k}', append=True)
            if self.ignore_stop:
                stats = stats.loc[stats.index.get_level_values(f'aa{k}') != '*']

        if normed:
            stats = stats / stats.groupby(aa_levels).sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]
        if fillna:
            norm = stats.groupby(aa_levels).size().to_frame('deg').join(stats)
            if type(stats) == pd.DataFrame:
                stats = stats.apply(lambda x: x.fillna(1 / norm['deg']))
            else:
                stats = stats.fillna(1 / norm['deg'])

        stats = stats.reorder_levels(aa_levels + cod_levels).sort_index()
        if self.k_mer == 1:
            stats = stats.rename_axis(index=['aa', 'codon'])

        return stats


class NucleotideCounter(object):
    """
    Nucleotide statistics for a single, or multiple DNA sequences.
    When the `k_mer` argument is provided, the counter will return
    dinucleotide (k_mer=2), trinucleotide (k_mer=3) statistics, etc.

    Parameters
    ----------
    seqs : str, or iterable of str
        DNA sequence, or an iterable of ones.
    k_mer : int, optional
        Determines the length of the k-mer to base statistics on, by
        default 1
    sum_seqs : bool, optional
        Determines how multiple sequences will be handled. When True,
        their statistics will be summed, otherwise separate statistics
        will be kept in a table. by default True
    """
    def __init__(self, seqs, k_mer=1, sum_seqs=True):
        self.k_mer = k_mer
        self.sum_seqs = sum_seqs

        self.counts = self._count(seqs)
        self.counts = self.counts.loc[
            self.counts.index.str.contains('^[ACGT]+$', regex=True)]\
            .sort_index()
        if self.counts.ndim == 1:
            self.counts = self.counts.rename('count')

    def _count(self, seqs):
        if isinstance(seqs, str):
            return self._count_single(seqs)

        counts = pd.concat([self._count_single(s) for s in seqs], axis=1)\
            .fillna(0)
        if self.sum_seqs:
            return counts.sum(axis=1)
        else:
            return counts

    def _count_single(self, seq):
        seq = seq.upper().replace('U', 'T')

        last_pos = len(seq) - self.k_mer + 1
        return pd.Series(Counter(
            [seq[i:i+self.k_mer] for i in range(last_pos)]))

    def get_table(self, normed=False):
        """
        Return nucleotide counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False),
        indexed by the nucletoide k-mer.

        Parameters
        ----------
        normed : bool, optional
            Determines whether nucleotide counts will be normalized to sum
            to 1, by default False

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Neltodie k-mer counts (or frequencies) with k-mers as index,
            and counts as values.
        """
        if normed:
            return self.counts / self.counts.sum()
        else:
            return self.counts
