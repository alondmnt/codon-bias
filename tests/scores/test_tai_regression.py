import pandas as pd

from codonbias.scores import TrnaAdaptationIndex


def test_tai_ecoli_regression(ecoli_seqs, ecoli_tgcn, dataframe_regression):
    """E. coli tAI scores. Reference captured before the NumPy conversion (#8)."""
    tai = TrnaAdaptationIndex(tGCN=ecoli_tgcn, prokaryote=True)
    scores = tai.get_score(ecoli_seqs)

    df = pd.DataFrame({"gene_index": range(len(scores)), "score": scores})
    df["score"] = df["score"].round(6)
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)
