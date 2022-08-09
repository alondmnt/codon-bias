from collections import Counter, Iterable
import os

import pandas as pd

gc = pd.read_csv(f'{os.path.dirname(__file__)}/genetic_code_ncbi.csv',
                 index_col=0).sort_index()
# https://en.wikipedia.org/wiki/List_of_genetic_codes


class CodonCounter(object):
    def __init__(self, seqs, genetic_code=1):
        self.genetic_code = str(genetic_code)
        self.counts = pd.Series(self._count(seqs), name='count')

    def _count(self, seqs):
        if not isinstance(seqs, str):
            return sum([self._count(s) for s in seqs], start=Counter())

        seqs = seqs.upper().replace('U', 'T')

        return Counter([seqs[i:i+3] for i in range(0, len(seqs), 3)])

    def get_codon_table(self, normed=False):
        stats = gc[[self.genetic_code]].join(self.counts)\
            .fillna(0.)['count']
        if normed:
            stats /= stats.sum()

        return stats

    def get_aa_table(self, normed=False):
        stats = gc[[self.genetic_code]].join(self.counts)\
            .rename(columns={self.genetic_code: 'aa'}).fillna(0.)\
            .set_index('aa', append=True).reorder_levels(['aa', 'codon'])\
            .sort_index()
        if normed:
            stats = stats.join(stats.groupby('aa').sum()\
                .rename(columns={'count': 'norm'}))
            stats['count'] /= stats['norm']

        return stats['count']
