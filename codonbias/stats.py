from collections import Counter
import os

import pandas as pd

gc = pd.read_csv(f'{os.path.dirname(__file__)}/genetic_code_ncbi.csv',
                 index_col=0).sort_index()
# https://en.wikipedia.org/wiki/List_of_genetic_codes

class CodonCounter(object):
    """
    Codon statistics for a single, or multiple DNA sequences.

    Parameters
    ----------
    seqs : str, or iterable of str
        DNA sequence, or an iterable of ones.
    sum_seqs : bool, optional
        Determines how multiple sequences will be handled. When True,
        their statistics will be summed, otherwise separate statistics
        will be kept in a table. by default True
    genetic_code : int, optional
        NCBI genetic code ID, by default 1
    ignore_stop : bool, optional
        Whether STOP codons will be discarded from the analysis, by default True
    """
    def __init__(self, seqs, sum_seqs=True, genetic_code=1, ignore_stop=True):
        self.sum_seqs = sum_seqs
        self.genetic_code = str(genetic_code)
        self.ignore_stop = ignore_stop
        self.counts = self._count(seqs)
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

        return pd.Series(Counter([seq[i:i+3] for i in range(0, len(seq), 3)]))

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
        stats = gc[[self.genetic_code]].join(self.counts)\
            .fillna(0.)
        if self.ignore_stop:
            stats = stats.loc[stats[self.genetic_code] != '*']
        stats = stats.drop(columns=self.genetic_code)
        if normed:
            stats /= stats.sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]
        if fillna:
            stats = stats.fillna(1/len(stats))

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
        stats = gc[[self.genetic_code]].join(self.counts)\
            .rename(columns={self.genetic_code: 'aa'}).fillna(0.)\
            .set_index('aa', append=True).reorder_levels(['aa', 'codon'])\
            .sort_index()
        if self.ignore_stop:
            stats = stats.loc[stats.index.get_level_values('aa') != '*']
        if normed:
            stats /= stats.groupby('aa').sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]
        if fillna:
            norm = stats.groupby('aa').size().to_frame('deg').join(stats)
            if type(stats) == pd.DataFrame:
                stats = stats.apply(lambda x: x.fillna(1 / norm['deg']))
            else:
                stats = stats.fillna(1 / norm['deg'])

        return stats
