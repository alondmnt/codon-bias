import os
from itertools import product

import numpy as np
import pandas as pd

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

        # Base-5 packed codon LUT for the vectorised path. Codon ids are
        # packed b0*25 + b1*5 + b2 so sentinel-containing triplets never
        # collide with valid ACGT triplets. Values are indices into
        # ``codon_index`` (-1 for stops or sentinel-containing triplets).
        self._codon_lex_to_idx = np.full(125, -1, dtype=np.int32)
        for codon_idx, codon in enumerate(self.codon_index):
            lex = (
                25 * "ACGT".index(codon[0])
                + 5 * "ACGT".index(codon[1])
                + "ACGT".index(codon[2])
            )
            self._codon_lex_to_idx[lex] = codon_idx

        # Per-codon base indices (A=0 C=1 G=2 T=3) for callers that need
        # to read background nucleotide compositions. Shape
        # (len(codon_index), 3) int8. Always built; k_mer>1 callers
        # consume it via ``codon_base_idx_kmer`` below.
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.codon_base_idx = np.array(
            [[base_to_idx[b] for b in cod] for cod in self.codon_index],
            dtype=np.int8,
        )

    def count_array(self, seq):
        """Stateless k-mer codon count.

        Returns an ndarray of shape ``(len(self.codon_index) ** k_mer,)``
        ordered by the lex product of ``self.codon_index`` (k_mer=1
        reduces to ``self.codon_index``). K-mers containing a stop or
        non-ACGT base are dropped, matching the observable output of
        ``get_codon_table``. Does not touch ``self.counts``.

        Supported for k_mer in [1, 3]; above that the dense aligned
        output would require >14M entries per call and the method
        raises ``NotImplementedError``.

        Parameters
        ----------
        seq : str
            DNA sequence.

        Returns
        -------
        numpy.ndarray
            Codon (or codon k-mer) counts as float.
        """
        if self.k_mer > 3:
            raise NotImplementedError(
                f"count_array supports k_mer <= 3 (got k_mer={self.k_mer}); "
                f"a dense aligned output would need "
                f"{len(self.codon_index) ** self.k_mer:,} entries"
            )
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")

        seq = seq.upper().replace("U", "T")
        b = seq.encode("ascii", errors="replace")
        n_codons_total = len(b) // 3
        arr = np.frombuffer(b[: n_codons_total * 3], dtype=np.uint8).reshape(
            n_codons_total, 3
        )
        base_ids = _BASE_LUT[arr]
        lex_ids = base_ids[:, 0] * 25 + base_ids[:, 1] * 5 + base_ids[:, 2]
        codon_ids = self._codon_lex_to_idx[lex_ids]

        n_codons = len(self.codon_index)
        n_out = n_codons**self.k_mer
        if self.k_mer == 1:
            valid = codon_ids >= 0
            return np.bincount(codon_ids[valid], minlength=n_out).astype(float)

        # k_mer in {2, 3}: sliding window over codon ids (stride 1, matching
        # iter_codons step=3), then combine each window into a single bucket
        # id ``sum(idx[i] * n_codons ** (k-1-i))`` for bincount. Windows that
        # contain a stop/sentinel codon are dropped, mirroring the k_mer=1
        # path.
        k = self.k_mer
        if n_codons_total < k:
            return np.zeros(n_out, dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(codon_ids, k)
        valid = (windows >= 0).all(axis=1)
        powers = n_codons ** np.arange(k - 1, -1, -1)
        combined = (windows * powers).sum(axis=1)
        return np.bincount(combined[valid], minlength=n_out).astype(float)

    def count(self, seqs):
        """
        Update the CodonCounter object with the codon counts of the given
        sequence(s).

        Routes through ``count_array`` for the heavy lifting, then wraps
        the resulting ndarray back into a pandas Series/DataFrame indexed
        by the lex-product codon order. The MultiIndex split (when
        ``concat_index=False``) is applied here as a presentation step.

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
        index = self.kmer_index
        self.counts = (
            pd.Series(res, index=index, name="count")
            if res.ndim == 1
            else pd.DataFrame(res, index=index)
        )
        self.counts.index.name = "codon"
        if self.k_mer > 1 and not self.concat_index:
            self.counts.index = pd.MultiIndex.from_arrays(
                [self.counts.index.str[3 * k : 3 * (k + 1)] for k in range(self.k_mer)],
                names=[f"codon{k}" for k in range(self.k_mer)],
            )
        return self

    def _count(self, seqs):
        if isinstance(seqs, str):
            return self.count_array(seqs)
        if isinstance(seqs, (list, np.ndarray)):
            counts = np.column_stack([self.count_array(s) for s in seqs])
            return counts.sum(axis=1) if self.sum_seqs else counts
        raise ValueError(f"unknown sequence type: {type(seqs)}")

    @property
    def kmer_index(self):
        """Concat-string index aligned to ``count_array``'s output order.

        For k_mer=1 this is just ``codon_index``; for k_mer>1 it is the
        lex product of ``codon_index`` joined into k-mer strings (e.g.,
        ``['AAAAAA', 'AAAACC', ...]`` for k_mer=2). Built lazily.
        """
        if self.k_mer == 1:
            return self.codon_index
        if not hasattr(self, "_kmer_index"):
            self._kmer_index = [
                "".join(t) for t in product(self.codon_index, repeat=self.k_mer)
            ]
        return self._kmer_index

    @property
    def aa_group_kmer(self):
        """Aa-tuple group id for each k-mer in lex-product order.

        Shape ``(len(codon_index) ** k_mer,)``. For k_mer=1 this is just
        ``aa_group``. For k_mer>1 the i-th codon position contributes
        ``aa_group[c_i] * n_aa ** (k - 1 - i)``, so the result indexes
        the cartesian product of aa groups across positions. Lets
        aa-grouped reductions (e.g., ENC's chi-square sum) work
        uniformly across k_mer via ``np.bincount(aa_group_kmer, ...)``.
        Built lazily.
        """
        if self.k_mer == 1:
            return self.aa_group
        if not hasattr(self, "_aa_group_kmer"):
            n_codons = len(self.codon_index)
            # idx[i] is shape (n_codons**k_mer,) and gives the codon index
            # at k-mer position i for every bucket in lex-product order.
            idx = np.indices((n_codons,) * self.k_mer).reshape(self.k_mer, -1)
            result = np.zeros(n_codons**self.k_mer, dtype=np.int32)
            for i in range(self.k_mer):
                result += self.aa_group[idx[i]] * self.n_aa ** (self.k_mer - 1 - i)
            self._aa_group_kmer = result
        return self._aa_group_kmer

    @property
    def codon_base_idx_kmer(self):
        """Per-k-mer base indices, shape ``(n_codons ** k_mer, 3 * k_mer)``.

        Generalises ``codon_base_idx`` to any k_mer: for each k-mer in
        lex-product order, the row holds the base indices at all
        ``3 * k_mer`` positions (concatenated across k-mer positions).
        Built lazily.
        """
        if self.k_mer == 1:
            return self.codon_base_idx
        if not hasattr(self, "_codon_base_idx_kmer"):
            n_codons = len(self.codon_index)
            idx = np.indices((n_codons,) * self.k_mer).reshape(self.k_mer, -1)
            self._codon_base_idx_kmer = np.concatenate(
                [self.codon_base_idx[idx[i]] for i in range(self.k_mer)], axis=1
            )
        return self._codon_base_idx_kmer

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
        """Stateless k-mer base count.

        Returns an ndarray of shape ``(4 ** k_mer,)`` ordered by the lex
        product of ACGT. Respects ``self.frame`` and ``self.step`` (which
        select the *starting* positions of each k-mer; the k bases inside
        each k-mer are always consecutive). K-mers containing any
        non-ACGT base are dropped. Does not touch ``self.counts``.

        Parameters
        ----------
        seq : str
            Nucleotide sequence.

        Returns
        -------
        numpy.ndarray
            Base (or base k-mer) counts as int.
        """
        if not isinstance(seq, str):
            raise ValueError(f"sequence is not a string: {type(seq)}")

        seq = seq.upper().replace("U", "T")
        b = seq.encode("ascii", errors="replace")
        arr = np.frombuffer(b, dtype=np.uint8)
        base_ids = _BASE_LUT[arr]

        k = self.k_mer
        n_out = 4**k
        if k == 1:
            strided = base_ids[self.frame - 1 :: self.step]
            return np.bincount(strided[strided < 4], minlength=n_out)

        # k_mer > 1: take all consecutive k-grams, then keep starts on the
        # ``frame``/``step`` grid (matches the original Counter generator
        # ``range(frame-1, last_pos, step)`` over ``seq[i:i+k]``).
        if len(base_ids) < k:
            return np.zeros(n_out, dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(base_ids, k)
        windows = windows[self.frame - 1 :: self.step]
        valid = (windows < 4).all(axis=1)
        powers = 4 ** np.arange(k - 1, -1, -1)
        combined = (windows * powers).sum(axis=1)
        return np.bincount(combined[valid], minlength=n_out)

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
        index = self.kmer_index
        self.counts = (
            pd.Series(res, index=index, name="count")
            if res.ndim == 1
            else pd.DataFrame(res, index=index)
        )
        return self

    def _count(self, seqs):
        if isinstance(seqs, str):
            return self.count_array(seqs)
        if isinstance(seqs, (list, np.ndarray)):
            counts = np.column_stack([self.count_array(s) for s in seqs])
            return counts.sum(axis=1) if self.sum_seqs else counts
        raise ValueError(f"unknown sequence type: {type(seqs)}")

    @property
    def kmer_index(self):
        """Concat-string index aligned to ``count_array``'s output order.

        For k_mer=1 this is ``['A', 'C', 'G', 'T']``; for k_mer>1 it is
        the lex product of ACGT joined into k-mer strings (e.g.,
        ``['AA', 'AC', ..., 'TT']`` for k_mer=2). Built lazily.
        """
        if self.k_mer == 1:
            return list("ACGT")
        if not hasattr(self, "_kmer_index"):
            self._kmer_index = ["".join(t) for t in product("ACGT", repeat=self.k_mer)]
        return self._kmer_index

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
