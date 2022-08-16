import numpy as np
import pandas as pd

complement = {'A': 'T',
              'C': 'G',
              'G': 'C',
              'T': 'A'}


def reverse_complement(seq):
    return ''.join([complement[b] for b in seq[::-1]])


def geomean(weights, counts):
    nn = weights.index[np.isfinite(np.log(weights))]
    return np.exp((np.log(weights[nn]) * counts.reindex(nn)).sum() / counts.reindex(nn).sum())


def mean(weights, counts):
    nn = weights.index[np.isfinite(weights)]
    return (weights[nn] * counts.reindex(nn)).sum() / counts.reindex(nn).sum()


def fetch_GCN_from_GtRNAdb(url=None, genome=None, domain=None):
    if genome is not None and domain is not None:
        url = f'http://gtrnadb.ucsc.edu/genomes/{domain}/{genome}/'
    tables = pd.read_html(url)

    return pd.concat(
        [process_GtRNAdb_table(t) for t in tables[2:]],
        axis=0, ignore_index=True).sort_values('anti_codon')


def process_GtRNAdb_table(table):
    """ get a dataframe of tRNA anti-codon copy numbers from a table. """
    df = table.loc[:, table.dtypes == object].apply(lambda col: col.str.split(' ').str[-2:])
    # flatten
    df = pd.DataFrame({'pair': df.values[df.apply(lambda col: col.str.len()).values == 2]})
    # rearrange
    df['anti_codon'] = df['pair'].str[0]
    df['GCN'] = df['pair'].str[1].str.split('/').apply(lambda x: sum(map(int, x)))

    return df.drop(columns='pair')
