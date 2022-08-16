from collections import Counter
from itertools import product
import os

import numpy as np
import pandas as pd
from scipy import stats

from .stats import CodonCounter
from .utils import fetch_GCN_from_GtRNAdb, geomean, mean, reverse_complement


class ScalarScore(object):
    def __init__(self):
        pass

    def get_score(self, seq, slice=None):
        if not isinstance(seq, str):
            return np.array([self.get_score(s, slice=slice) for s in seq])

        if slice is not None:
            return self._calc_score(seq[slice])
        else:
            return self._calc_score(seq)

    def _calc_score(self, seq):
        raise Exception('not implemented')


class VectorScore(object):
    def __init__(self):
        pass

    def get_vector(self, seq, slice=None):
        if not isinstance(seq, str):
            return np.array([self.get_vector(s, slice=slice) for s in seq])

        if slice is not None:
            return self._calc_vector(seq[slice])
        else:
            return self._calc_vector(seq)

    def _calc_vector(self, seq):
        raise Exception('not implemented')

    def _get_codon_vector(self, seq):
        return [seq[i:i+3] for i in range(0, len(seq), 3)]


class FrequencyOfOptimalCodons(ScalarScore, VectorScore):
    def __init__(self, ref_seq, thresh=0.95, genetic_code=1, ignore_stop=True):
        """ Ikemura, J Mol Biol 1981 """
        self.thresh = thresh
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

        self.weights = CodonCounter(ref_seq,
            genetic_code=genetic_code, ignore_stop=ignore_stop)\
            .get_aa_table().groupby('aa').apply(lambda x: x / x.max())
        self.weights[self.weights >= self.thresh] = 1  # optimal
        self.weights[self.weights < self.thresh] = 0  # non-optimal
        self.weights = self.weights.droplevel('aa')

    def _calc_score(self, seq):
        counts = CodonCounter(seq, self.genetic_code).counts

        return mean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.loc[self._get_codon_vector(seq)].values


class RelativeSynonymousCodonUsage(VectorScore):
    def __init__(self, ref_seq=None, genetic_code=1, ignore_stop=True):
        """ Sharp & Li, NAR 1986 """
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

        if ref_seq is None:
            ref = CodonCounter('',
                genetic_code=genetic_code, ignore_stop=ignore_stop)\
                .get_aa_table().to_frame('count')
            ref = ref.join(ref.groupby('aa').size().to_frame('deg'))
            ref['count'] = 1/ref['deg']
            self.reference = ref['count']
        else:
            self.reference = CodonCounter(ref_seq,
                genetic_code=genetic_code, ignore_stop=ignore_stop)\
                .get_aa_table(normed=True)

    def _calc_vector(self, seq):
        counts = CodonCounter(seq,
            genetic_code=self.genetic_code, ignore_stop=self.ignore_stop)\
            .get_aa_table(normed=True)
        return counts / self.reference


class CodonAdaptationIndex(ScalarScore, VectorScore):
    def __init__(self, ref_seq, genetic_code=1, ignore_stop=True):
        """ Sharp & Li, NAR 1987 """
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

        self.weights = CodonCounter(ref_seq,
            genetic_code=genetic_code, ignore_stop=ignore_stop)\
            .get_aa_table().groupby('aa').apply(lambda x: x / x.max())
        self.weights = self.weights.droplevel('aa')

    def _calc_score(self, seq):
        counts = CodonCounter(seq, self.genetic_code).counts

        return geomean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.loc[self._get_codon_vector(seq)].values


class EffectiveNumberOfCodons(ScalarScore):
    def __init__(self, genetic_code=1):
        """ Wright, Gene 1990 """
        self.genetic_code = genetic_code
        self.ignore_stop = True  # score is not defined for STOP codons

    def _calc_score(self, seq):
        counts = CodonCounter(seq,
            genetic_code=self.genetic_code, ignore_stop=self.ignore_stop)\
            .get_aa_table()

        N = counts.groupby('aa').sum()
        P = counts / N
        F = ((N * (P**2).groupby('aa').sum() - 1) / (N-1)).to_frame('F')
        F['deg'] = P.groupby('aa').size()
        deg_count = F.groupby('deg').size().to_frame('deg_count')

        # at least 2 samples from AA to be included
        F = F.loc[(N > 1) & (F['F'] > 0)].groupby('deg').mean()\
            .join(deg_count, how='right')

        # misssing AA cases
        miss_3 = np.isnan(F.loc[3, 'F'])
        F['F'] = F['F'].fillna(1/F.index.to_series())  # use 1/deg
        if miss_3:
            F.loc[3, 'F'] = 0.5*(F.loc[2, 'F'] + F.loc[4, 'F'])

        ENC = (F['deg_count'] / F['F']).sum()
        return min([61., ENC])


class TrnaAdaptationIndex(ScalarScore, VectorScore):
    def __init__(self, tGCN=None, url=None, genome_id=None, domain=None,
                 prokaryote=False, s_values='dosReis', genetic_code=1):
        """ dos Reis, Savva & Wernisch, NAR 2004. """
        self.ignore_stop = True  # score is not defined for STOP codons
        self.genetic_code = genetic_code

        # tRNA gene copy numbers of the organism
        if url is not None or (genome_id is not None and domain is not None):
            tGCN = fetch_GCN_from_GtRNAdb(url=url, domain=domain, genome=genome_id)
        if tGCN is None:
            raise Exception('must provide either: tGCN dataframe, GtRNAdb url, or GtRNAdb genome_id+domain')
        tGCN['anti_codon'] = tGCN['anti_codon'].str.upper().str.replace('U', 'T')
        self.tGCN = tGCN

        # S-values: tRNA-codon efficiency of coupling
        self.s_values = pd.read_csv(
            f'{os.path.dirname(__file__)}/tAI_svalues_{s_values}.csv',
            dtype={'weight': float, 'prokaryote': bool}, comment='#')
        self.s_values['anti'] = self.s_values['anti'].str.upper().str.replace('U', 'T')
        self.s_values['cod'] = self.s_values['cod'].str.upper().str.replace('U', 'T')
        if not prokaryote:
            self.s_values = self.s_values.loc[~self.s_values['prokaryote']]

        self.weights = self._calc_weights()

    def _calc_weights(self):
        # init the dataframe
        weights = CodonCounter('',
            genetic_code=self.genetic_code, ignore_stop=self.ignore_stop)\
            .get_aa_table().to_frame('count')
        weights = weights.join(weights.groupby('aa').size().to_frame('deg'))\
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
        counts = CodonCounter(seq, self.genetic_code).counts

        return geomean(self.weights, counts)

    def _calc_vector(self, seq):
        return self.weights.loc[self._get_codon_vector(seq)].values


class RelativeCodonBiasScore(ScalarScore, VectorScore):
    def __init__(self, directional=False, genetic_code=1, ignore_stop=True):
        """ Roymondal, Das & Sahoo, DNA Research 2009.
            directional: Sabi & Tuller, DNA Research 2014. """
        self.directional = directional
        self.genetic_code = genetic_code
        self.ignore_stop = ignore_stop

    def _calc_score(self, seq):
        counts = CodonCounter(seq, self.genetic_code, self.ignore_stop).counts
        D = self._calc_weights(seq)

        if self.directional:
            return mean(D, counts)
        else:
            return geomean(1 + D, counts) - 1

    def _calc_vector(self, seq):
        D = self._calc_weights(seq)

        return D.loc[self._get_codon_vector(seq)].values

    def _calc_weights(self, seq):
        counts = CodonCounter(seq, self.genetic_code, self.ignore_stop)
        # background probabilities
        BCC = self._calc_BCC(self._calc_BNC(seq))
        # observed probabilities
        P = counts.get_codon_table(normed=True)
        # codon weights
        if self.directional:
            D = np.maximum(P / BCC, BCC / P)
        else:
            D = (P - BCC) / BCC

        return D

    def _calc_BNC(self, seq):
        """ calculate the background NUCLEOTIDE composition of the sequence. """
        BNC = pd.concat([pd.Series(Counter(seq[i::3])) for i in range(3)], axis=1)

        return BNC

    def _calc_BCC(self, BNC):
        """ calculate the background CODON composition of the sequence. """
        BCC = pd.DataFrame(
            [(c1+c2+c3, BNC[0][c1] * BNC[1][c2] * BNC[2][c3])
             for c1, c2, c3 in product('ACGT', 'ACGT', 'ACGT')],
            columns=['codon', 'bcc'])
        BCC = BCC.set_index('codon')['bcc']
        BCC /= BCC.sum()

        return BCC 
