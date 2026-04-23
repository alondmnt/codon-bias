import pandas as pd
import pytest

from codonbias.scores import CodonAdaptationIndex


@pytest.mark.parametrize("k_mer", [1, 2], ids=["kmer1", "kmer2"])
def test_cai_ecoli_regression(ecoli_seqs, dataframe_regression, k_mer):
    """E. coli CAI scores. Reference captured before the NumPy conversion (#8).

    k_mer=1 exercises the new NumPy path; k_mer=2 exercises the unchanged
    Series path — both must match the pre-conversion reference.
    """
    cai = CodonAdaptationIndex(ref_seq=ecoli_seqs, k_mer=k_mer)
    scores = cai.get_score(ecoli_seqs)

    df = pd.DataFrame({"gene_index": range(len(scores)), "score": scores})
    df["score"] = df["score"].round(6)
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)
