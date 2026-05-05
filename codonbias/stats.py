import os
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from .utils import iter_codons

gc = pd.read_csv(
    f"{os.path.dirname(__file__)}/genetic_code_ncbi.csv", index_col=0
).sort_index()

# Byte-level LUT shared by the vectorised k_mer=1 paths.
# A=0, C=1, G=2, T=3, sentinel=4 for any other byte.
_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _b, _i in zip(b"ACGT", range(4)):
    _BASE_LUT[_b] = _i


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

    def __init__(
        self,
        seqs=None,
        k_mer=1,
        sum_seqs=True,
        concat_index=True,
        genetic_code=1,
        ignore_stop=True,
    ):
        self.k_mer = k_mer
        self.concat_index = concat_index
        self.sum_seqs = sum_seqs
        self.genetic_code = str(genetic_code)
        self.ignore_stop = ignore_stop

        self._init_arrays()

        if seqs is not None:
            self.count(seqs)

    def _init_arrays(self):
        code = (
            gc[[self.genetic_code]]
            .reset_index()
            .rename(columns={self.genetic_code: "aa", "index": "codon"})
        )
        if self.ignore_stop:
            code = code.loc[code["aa"] != "*"]

        code = code.sort_values(["aa", "codon"])

        self.codon_index = code["codon"].tolist()
        self._codon_to_idx = {c: i for i, c in enumerate(self.codon_index)}

        unique_aa = code["aa"].unique()
        self.n_aa = len(unique_aa)
        self._aa_to_idx = {aa: i for i, aa in enumerate(unique_aa)}
        self.aa_group = np.array([self._aa_to_idx[aa] for aa in code["aa"]])

        # Transit aliases for the renamed attrs. Removed once all callers
        # migrate to the public names.
        self._idx_to_codon = self.codon_index
        self._n_aa = self.n_aa
        self._aa_group = self.aa_group

        # Base-5 packed codon LUT for the vectorised k_mer=1 path. Codon
        # ids are packed b0*25 + b1*5 + b2 so sentinel-containing triplets
        # never collide with valid ACGT triplets.
        self._codon_lex_to_aa = np.full(125, -1, dtype=np.int32)
        for aa_idx, codon in enumerate(self.codon_index):
            lex = (
                25 * "ACGT".index(codon[0])
                + 5 * "ACGT".index(codon[1])
                + "ACGT".index(codon[2])
            )
            self._codon_lex_to_aa[lex] = aa_idx

    def count_array(self, seq):
        """Stateless k_mer=1 codon count.

        Returns an ndarray of shape ``(len(self.codon_index),)`` ordered
        by ``self.codon_index``. Does not touch ``self.counts``.

        Parameters
        ----------
        seq : str
            DNA sequence.

        Returns
        -------
        numpy.ndarray
            Codon counts as float.
        """
        if self.k_mer != 1:
            raise NotImplementedError("count_array is currently k_mer=1 only")
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")

        seq = seq.upper().replace("U", "T")
        b = seq.encode("ascii", errors="replace")
        n_codons = len(b) // 3
        arr = np.frombuffer(b[: n_codons * 3], dtype=np.uint8).reshape(n_codons, 3)
        base_ids = _BASE_LUT[arr]
        lex_ids = base_ids[:, 0] * 25 + base_ids[:, 1] * 5 + base_ids[:, 2]
        aa_ids = self._codon_lex_to_aa[lex_ids]
        valid = aa_ids >= 0
        return np.bincount(aa_ids[valid], minlength=len(self.codon_index)).astype(float)

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
        res = self._count(seqs)
        # MINIMAL CHANGE: Wrap NumPy back to Pandas at the boundary for k_mer=1
        if self.k_mer == 1:
            self.counts = (
                pd.Series(res, index=self.codon_index, name="count")
                if res.ndim == 1
                else pd.DataFrame(res, index=self.codon_index)
            )
            self.counts.index.name = "codon"
        else:
            self.counts = self._format_counts(res)
            if self.counts.ndim == 1:
                self.counts = self.counts.rename("count")
        return self

    def _count(self, seqs):
        if isinstance(seqs, str):
            res = self._count_single(seqs)
            return (
                res[0] if self.k_mer == 1 else res
            )  # Return only count array for k_mer=1
        elif isinstance(seqs, (list, np.ndarray)):
            if self.k_mer == 1:
                counts = np.column_stack([self._count_single(s)[0] for s in seqs])
            else:
                counts = pd.concat(
                    [self._count_single(s) for s in seqs], axis=1
                ).fillna(0)
            return counts.sum(axis=1) if self.sum_seqs else counts
        raise ValueError(f"unknown sequence type: {type(seqs)}")

    def _count_single(self, seq):
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")
        seq = seq.upper().replace("U", "T")

        if self.k_mer == 1:
            b = seq.encode("ascii", errors="replace")
            n_codons = len(b) // 3
            arr = np.frombuffer(b[: n_codons * 3], dtype=np.uint8).reshape(n_codons, 3)
            base_ids = _BASE_LUT[arr]
            lex_ids = base_ids[:, 0] * 25 + base_ids[:, 1] * 5 + base_ids[:, 2]
            aa_ids = self._codon_lex_to_aa[lex_ids]
            valid = aa_ids >= 0
            counts = np.bincount(aa_ids[valid], minlength=len(self.codon_index)).astype(
                float
            )
            aa_counts = np.bincount(self.aa_group, weights=counts, minlength=self.n_aa)
            return counts, aa_counts

        return pd.Series(
            Counter(iter_codons(seq, k_mer=self.k_mer)),
            dtype=int,
        )

    def _format_counts(self, counts):
        counts.index.name = "codon"

        if self.concat_index:
            return counts

        counts.index = pd.MultiIndex.from_arrays(
            [counts.index.str[3 * k : 3 * (k + 1)] for k in range(self.k_mer)],
            names=[f"codon{k}" for k in range(self.k_mer)],
        )

        return counts

    def get_codon_table(self, normed=False, pseudocount=1, nonzero=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when sum_seqs is False).
        Normalized frequencies (when normed=True) are corrected by
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
        if not hasattr(self, "template_cod"):
            self.template_cod = self._init_table(keep_aa=False)

        stats = self.template_cod.join(self.counts).fillna(0).drop(columns=["dummy"])

        if nonzero:
            stats = stats[stats != 0].dropna(how="all")
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
            stats = stats.rename_axis(index="codon")

        return stats

    def get_aa_table(self, normed=False, pseudocount=1, nonzero=False):
        """
        Return codon counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when sum_seqs is False),
        indexed by the codon and the encoded amino acid. Normalized
        frequencies (when normed=True) are corrected by default using
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
        if not hasattr(self, "template_aa"):
            self.template_aa = self._init_table(keep_aa=True)

        levels = self.template_aa.index.names
        stats = self.template_aa.join(self.counts).fillna(0).drop(columns=["dummy"])

        if nonzero:
            stats = stats[stats != 0].dropna(how="all")
        if normed:
            if pseudocount:
                stats += pseudocount
            aa_levels = [l for l in levels if "aa" in l]
            stats = stats / stats.groupby(aa_levels).sum()
        if stats.shape[1] == 1:
            stats = stats.iloc[:, 0]

        stats = stats.reorder_levels(levels).sort_index()
        if self.k_mer == 1:
            stats = stats.rename_axis(index=["aa", "codon"])

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
        code = (
            gc[[self.genetic_code]]
            .reset_index()
            .assign(dummy=1)
            .rename(columns={self.genetic_code: "aa"})
        )
        if self.ignore_stop:
            code = code.loc[code["aa"] != "*"]
        levels = ["aa", "codon"]
        if not keep_aa:
            code = code.drop(columns=["aa"])
            levels = "codon"

        if self.k_mer == 1:
            return code.set_index(levels)

        stats = code.rename(columns={"codon": "codon0", "aa": "aa0"}, errors="ignore")

        for k in range(self.k_mer - 1):
            stats = code.rename(
                columns={"codon": f"codon{k + 1}", "aa": f"aa{k + 1}"}, errors="ignore"
            ).merge(stats, on="dummy", how="left")

        aa_levels = [f"aa{k}" for k in range(self.k_mer) if keep_aa]
        cod_levels = [f"codon{k}" for k in range(self.k_mer)]
        stats = stats.set_index(aa_levels + cod_levels)

        if not self.concat_index:
            return stats

        if keep_aa:
            stats.index = pd.MultiIndex.from_arrays(
                [
                    stats.index.droplevel(cod_levels).to_series().str.join(""),
                    stats.index.droplevel(aa_levels).to_series().str.join(""),
                ],
                names=["aa", "codon"],
            )
        else:
            stats.index = stats.index.to_series().str.join("")
            stats.index.name = "codon"

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

    def count_array(self, seq):
        """Stateless k_mer=1 base count.

        Returns an ndarray of shape ``(4,)`` in ACGT order. Respects
        ``self.frame`` and ``self.step``. Does not touch ``self.counts``.

        Parameters
        ----------
        seq : str
            Nucleotide sequence.

        Returns
        -------
        numpy.ndarray
            Base counts as int.
        """
        if self.k_mer != 1:
            raise NotImplementedError("count_array is currently k_mer=1 only")
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")

        seq = seq.upper().replace("U", "T")
        b = seq.encode("ascii", errors="replace")
        arr = np.frombuffer(b, dtype=np.uint8)[self.frame - 1 :: self.step]
        base_ids = _BASE_LUT[arr]
        return np.bincount(base_ids[base_ids < 4], minlength=4)

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
        res = self._count(seqs)
        if self.k_mer == 1:
            # _count returns ndarray: (4,) for single/sum, (4, N) for multi.
            index = list("ACGT")
            self.counts = (
                pd.Series(res, index=index, name="count")
                if res.ndim == 1
                else pd.DataFrame(res, index=index)
            )
        else:
            self.counts = res.reindex(self._init_table()).fillna(0)
            if self.counts.ndim == 1:
                self.counts = self.counts.rename("count")

        return self

    def _count(self, seqs):
        if isinstance(seqs, str):
            return self._count_single(seqs)
        elif isinstance(seqs, (list, np.ndarray)):
            if self.k_mer == 1:
                counts = np.column_stack([self._count_single(s) for s in seqs])
            else:
                counts = pd.concat(
                    [self._count_single(s) for s in seqs], axis=1
                ).fillna(0)
            return counts.sum(axis=1) if self.sum_seqs else counts
        raise ValueError(f"unknown sequence type: {type(seqs)}")

    def _count_single(self, seq):
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")
        seq = seq.upper().replace("U", "T")

        if self.k_mer == 1:
            b = seq.encode("ascii", errors="replace")
            arr = np.frombuffer(b, dtype=np.uint8)[self.frame - 1 :: self.step]
            base_ids = _BASE_LUT[arr]
            return np.bincount(base_ids[base_ids < 4], minlength=4)

        last_pos = len(seq) - self.k_mer + 1
        return pd.Series(
            Counter(
                [
                    seq[i : i + self.k_mer]
                    for i in range(self.frame - 1, last_pos, self.step)
                ]
            ),
            dtype=int,
        )

    def get_table(self, normed=False, pseudocount=1):
        """
        Return base counts as a Series (for a single summary) or
        DataFrame (for multiple summaries, when sum_seqs is False),
        indexed by the nucletoide k-mer. Normalized frequencies (when
        normed=True) are corrected by default using pseudocounts.

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
        return ["".join(comb) for comb in list(product(*(self.k_mer * ["ACGT"])))]
