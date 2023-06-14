from collections import Counter
from itertools import product
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
    seqs : str, or iterable of str, optional
        DNA sequence, or an iterable of ones. by default None
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
    def __init__(self, seqs=None, k_mer=1, sum_seqs=True, concat_index=True,
                 genetic_code=1, ignore_stop=True):
        self.k_mer = k_mer
        self.concat_index = concat_index
        self.sum_seqs = sum_seqs
        self.genetic_code = str(genetic_code)
        self.ignore_stop = ignore_stop
        if seqs is not None:
            self.count(seqs)

    def count(self, seqs):
        """
        Update the CodonCounter object with the codon counts of the given
        sequence(s).

        Parameters
        ----------
        seqs : str, or iterable of str
            DNA sequence, or an iterable of ones. by default None

        Returns
        -------
        CodonCounter
            CodonCounter object (self) with updated counts
        """
        self.counts = self._format_counts(self._count(seqs))
        if self.counts.ndim == 1:
            self.counts = self.counts.rename('count')

        return self

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
             for i in range(0, len(seq), 3)]), dtype=int)

    def _format_counts(self, counts):
        counts.index.name = 'codon'

        if self.concat_index:
            return counts

        counts.index = pd.MultiIndex.from_arrays([
            counts.index.str[3*k:3*(k+1)]
            for k in range(self.k_mer)],
            names=[f'codon{k}' for k in range(self.k_mer)])

        return counts

    def get_codon_table(self, normed=False, pseudocount=1, nonzero=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False).
        Normalized frequencies (when `normed`=True) are corrected by
        default using pseudocounts.

        Parameters
        ----------
        normed : bool, optional
            Determines whether codon counts will be normalized to sum to
            1, by default False
        pseudocount : int, optional
            Pseudocount correction for normalized codon frequencies, by
            default 1

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Codon counts (or frequencies) with codons as index, and counts
            as values.
        """
        if not hasattr(self, 'template_cod'):
            self.template_cod = self._init_table(keep_aa=False)

        stats = self.template_cod.join(self.counts)\
            .fillna(0).drop(columns=['dummy'])

        if nonzero:
            stats = stats[stats != 0].dropna(how='all')
        if normed:
            if pseudocount:
                stats += pseudocount
            stats = stats / stats.sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]

        if not self.concat_index and (self.k_mer > 1):
            stats = stats.reorder_levels(self.template_cod.index.names)
        stats = stats.sort_index()
        if self.k_mer == 1:
            stats = stats.rename_axis(index='codon')

        return stats

    def get_aa_table(self, normed=False, pseudocount=1, nonzero=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False),
        indexed by the codon and the encoded amino acid. Normalized
        frequencies (when `normed`=True) are corrected by default using
        pseudocounts.

        Parameters
        ----------
        normed : bool, optional
            Determines whether codon counts will be normalized to sum to
            1 for each amino acid (a vector that sums to 20), by default
            False
        pseudocount : int, optional
            Pseudocount correction for normalized codon frequencies, by
            default 1

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Codon counts (or frequencies) with amino acids and codons as
            index, and counts as values.
        """
        if not hasattr(self, 'template_aa'):
            self.template_aa = self._init_table(keep_aa=True)

        levels = self.template_aa.index.names
        stats = self.template_aa.join(self.counts)\
            .fillna(0).drop(columns=['dummy'])

        if nonzero:
            stats = stats[stats != 0].dropna(how='all')
        if normed:
            if pseudocount:
                stats += pseudocount
            aa_levels = [l for l in levels if 'aa' in l]
            stats = stats / stats.groupby(aa_levels).sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]

        stats = stats.reorder_levels(levels).sort_index()
        if self.k_mer == 1:
            stats = stats.rename_axis(index=['aa', 'codon'])

        return stats

    def _init_table(self, keep_aa=True):
        """
        Helper function for initializing codon table templates. The
        template will contain all possible combinations of k-mer codons
        as its index.

        Parameters
        ----------
        keep_aa : bool, optional
            Whether to include amino acids in the index, by default True

        Returns
        -------
        pandas.DataFrame
            A dataframe with a `dummy` column
        """
        code = gc[[self.genetic_code]].reset_index().assign(dummy=1)\
            .rename(columns={self.genetic_code: 'aa'})
        if self.ignore_stop:
            code = code.loc[code['aa'] != '*']
        levels = ['aa', 'codon']
        if not keep_aa:
            code = code.drop(columns=['aa'])
            levels = 'codon'

        if self.k_mer == 1:
            return code.set_index(levels)

        stats = code.rename(columns={'codon': f'codon0',
                                     'aa': f'aa0'},
                            errors='ignore')

        for k in range(self.k_mer - 1):
            stats = code.rename(columns={'codon': f'codon{k+1}',
                                         'aa': f'aa{k+1}'},
                                errors='ignore')\
                .merge(stats, on='dummy', how='left')

        aa_levels = [f'aa{k}' for k in range(self.k_mer) if keep_aa]
        cod_levels = [f'codon{k}' for k in range(self.k_mer)]
        stats = stats.set_index(aa_levels + cod_levels)

        if not self.concat_index:
            return stats

        if keep_aa:
            stats.index = pd.MultiIndex.from_arrays(
                [stats.index.droplevel(cod_levels).to_series().str.join(''),
                 stats.index.droplevel(aa_levels).to_series().str.join('')],
                names=['aa', 'codon'])
        else:
            stats.index = stats.index.to_series().str.join('')
            stats.index.name = 'codon'

        return stats


class BaseCounter(object):
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
    step : int, optional
        Determines the step size to take along the sequence, by default
        1
    frame : int, optional
        Determines the frame, or shift+1, from the beginning of the
        sequence, by default 1
    sum_seqs : bool, optional
        Determines how multiple sequences will be handled. When True,
        their statistics will be summed, otherwise separate statistics
        will be kept in a table. by default True

    Examples
    --------
    Compute the GC3 content (GC in the third position of codons):

    >>> nuc = BaseCounter(step=3, frame=3)
    >>> freq = nuc.count(seq).get_table(normed=True)
    >>> freq['G'] + freq['C']

    Compute CpG content:

    >>> nuc = BaseCounter(k_mer=2)
    >>> freq = nuc.count(seq).get_table(normed=True)
    >>> freq['CG']
    """
    def __init__(self, seqs=None, k_mer=1, step=1, frame=1, sum_seqs=True):
        self.k_mer = k_mer
        self.step = step
        self.frame = frame
        self.sum_seqs = sum_seqs
        if seqs is not None:
            self.count(seqs)

    def count(self, seqs):
        """
        Update the BaseCounter object with the base counts of the given
        sequence(s).

        Parameters
        ----------
        seqs : str, or iterable of str
            DNA sequence, or an iterable of ones. by default None

        Returns
        -------
        BaseCounter
            BaseCounter object (self) with updated counts
        """
        self.counts = self._count(seqs)
        self.counts = self.counts.reindex(self._init_table()).fillna(0)
        if self.counts.ndim == 1:
            self.counts = self.counts.rename('count')

        return self

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
            [seq[i:i+self.k_mer]
             for i in range(self.frame-1, last_pos, self.step)]), dtype=int)

    def get_table(self, normed=False, pseudocount=1):
        """
        Return base counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when `sum_seqs` is False),
        indexed by the nucletoide k-mer. Normalized frequencies (when
        `normed`=True) are corrected by default using pseudocounts.

        Parameters
        ----------
        normed : bool, optional
            Determines whether base counts will be normalized to sum
            to 1, by default False
        pseudocount : int, optional
            Pseudocount correction for normalized base frequencies,
            by default 1

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Neltodie k-mer counts (or frequencies) with k-mers as index,
            and counts as values.
        """
        stats = self.counts.copy()
        if normed:
            if pseudocount:
                stats += pseudocount
            return stats / stats.sum()
        else:
            return stats

    def _init_table(self):
        return [''.join(comb) 
                for comb in list(product(*(self.k_mer*['ACGT'])))]
