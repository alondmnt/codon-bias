import numpy as np
import pandas as pd

complement = {'A': 'T',
              'C': 'G',
              'G': 'C',
              'T': 'A'}


def reverse_complement(seq):
    """
    The reverse complement of the given DNA sequence, such as the
    anti-codon that perfectly pairs with a codon.

    Parameters
    ----------
    seq : str
        Nucleotide sequence in {A,C,G,T}.

    Returns
    -------
    str
       The reverse complement sequence in {A,C,G,T}.
    """
    return ''.join([complement[b] for b in seq[::-1]])


def geomean(log_weights, counts):
    """
    Compute the geometric mean based on codon scores given in
    `log_weights` (weights in logarithmic scale), and codon counts give
    in `counts`.

    Parameters
    ----------
    log_weights : pandas.Series
        Codon scores in logarithmic scale, with codons as index and scores
        as values.
    counts : pandas.Series
        Codon counts, with codons as index and counts as values.

    Returns
    -------
    float
        Geometric mean.
    """
    nn = log_weights.index[np.isfinite(log_weights)]
    return np.exp((log_weights[nn] * counts.reindex(nn)).sum() / counts.reindex(nn).sum())


def mean(weights, counts):
    """
    Compute the arithmetic mean based on codon scores given in
    `weights`, and codon counts given in `counts`.

    Parameters
    ----------
    weights : pandas.Series
        Codon scores, with codons as index and scores as values.
    counts : pandas.Series
        Codon counts, with codons as index and counts as values.

    Returns
    -------
    float
        Arithmetic mean.
    """
    nn = weights.index[np.isfinite(weights)]
    return (weights[nn] * counts.reindex(nn)).sum() / counts.reindex(nn).sum()


def fetch_GCN_from_GtRNAdb(url=None, genome=None, domain=None):
    """
    Download a tRNA gene copy number (GCN) table for an organism
    from GtRNAdb, given either the URL of the relevant page, or the
    genome ID and taxonomic domain of the organism.
    Note, that this is an experimental function.

    Parameters
    ----------
    url : str, optional
        URL of the relevant page on GtRNAdb, by default None
    genome : str, optional
        Genome ID of the organism, by default None
    domain : str, optional
        Taxonomic domain of the organism, by default None

    Returns
    -------
    pandas.DataFrame
        tRNA gene copy numbers with the columns: `anti_codon`, `GCN`.

    Examples
    --------
    >>> fetch_GCN_from_GtRNAdb(url='http://gtrnadb.ucsc.edu/genomes/eukaryota/Scere3/')
    anti_codon  GCN
    10        AAC   14
    35        AAT   13
    17        ACG    6
    13        AGA   11
    ....

    >>> fetch_GCN_from_GtRNAdb(genome='Scere3', domain='eukaryota')
    anti_codon  GCN
    10        AAC   14
    35        AAT   13
    17        ACG    6
    13        AGA   11
    ....

    """
    if genome is not None and domain is not None:
        url = f'http://gtrnadb.ucsc.edu/genomes/{domain}/{genome}/'
    tables = pd.read_html(url)

    return pd.concat(
        [process_GtRNAdb_table(t) for t in tables[-4:]],
        axis=0, ignore_index=True).sort_values('anti_codon')


def process_GtRNAdb_table(table):
    """
    Helper function to get a dataframe of tRNA anti-codon copy numbers
    from a single HTML table.

    Parameters
    ----------
    table : pandas.DataFrame
        The product of read_html().

    Returns
    -------
    pandas.DataFrame
        tRNA gene copy numbers with the columns: `anti_codon`, `GCN`.
    """
    df = table.loc[:, table.dtypes == object].apply(lambda col: col.str.split(' ').str[-2:])
    # flatten
    df = pd.DataFrame({'pair': df.values[df.apply(lambda col: col.str.len()).values == 2]})
    # rearrange
    df['anti_codon'] = df['pair'].str[0]
    df['GCN'] = df['pair'].str[1].str.split('/').apply(lambda x: sum(map(int, x)))

    return df.drop(columns='pair')
