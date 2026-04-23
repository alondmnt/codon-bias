import pandas as pd

from codonbias.scores import FrequencyOfOptimalCodons


def test_fop_ecoli_regression(ecoli_seqs, dataframe_regression):
    """E. coli FOP scores. Reference captured before the NumPy conversion (#8)."""
    fop = FrequencyOfOptimalCodons(ref_seq=ecoli_seqs)
    scores = fop.get_score(ecoli_seqs)

    df = pd.DataFrame({"gene_index": range(len(scores)), "score": scores})
    df["score"] = df["score"].round(6)
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)
