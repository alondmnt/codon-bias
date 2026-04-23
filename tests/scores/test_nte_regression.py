import pandas as pd

from codonbias.scores import NormalizedTranslationalEfficiency


def test_nte_ecoli_regression(
    ecoli_seqs, ecoli_tgcn, ecoli_mrna_counts, dataframe_regression
):
    """E. coli nTE scores. Reference captured before the NumPy conversion (#8)."""
    nte = NormalizedTranslationalEfficiency(
        ref_seq=ecoli_seqs,
        mRNA_counts=ecoli_mrna_counts,
        tGCN=ecoli_tgcn,
        prokaryote=True,
    )
    scores = nte.get_score(ecoli_seqs)

    df = pd.DataFrame({"gene_index": range(len(scores)), "score": scores})
    df["score"] = df["score"].round(6)
    df = df.sort_values(by=["gene_index"]).reset_index(drop=True)

    dataframe_regression.check(df)
