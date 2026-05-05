import gzip
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from codonbias.scores import EffectiveNumberOfCodons

EXPECTED_MD5 = "aaee0253df6f7d1df7df00e84d582fd4"

# Default subset for CI: deterministic head slice of the parsed ORFeome.
# The full set runs the slowest ENC k_mer=2 regression in ~10 min; the
# subset brings it well under a minute while still exercising every
# parameter combination. Set `ECOLI_FULL=1` to load the full set (note:
# regression baselines are committed against the subset, so full mode
# also requires `--force-regen` for any test that uses pytest-regressions).
ECOLI_SUBSET_SIZE = 500


def get_file_md5(file_path):
    """Calculates the MD5 sum of a file efficiently by reading in chunks."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in 4K chunks to avoid loading large files entirely into memory
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


@pytest.fixture(scope="session")
def ecoli_seqs():
    """
    Parses the local E. coli K-12 coding sequences.
    Scope is 'session' so it only runs once per pytest invocation.
    Includes MD5 integrity checking to ensure the local file isn't corrupted.

    By default returns a deterministic head slice of the first
    ``ECOLI_SUBSET_SIZE`` sequences. Set the ``ECOLI_FULL=1`` environment
    variable to load the full set (used for one-off validation; regression
    baselines are committed against the subset).
    """
    # Look for the file in the .test_data folder relative to this script
    test_data_dir = os.path.join(os.path.dirname(__file__), ".test_data")
    file_path = os.path.join(test_data_dir, "ecoli_cds.fna.gz")

    # 1. Ensure the file actually exists locally
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing test data! Expected to find 'ecoli_cds.fna.gz' at:\n{file_path}\n"
            "Please ensure the file is saved in the correct directory."
        )

    # 2. Verify file integrity
    actual_md5 = get_file_md5(file_path)
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(
            f"\nMD5 mismatch for local genome file!\n"
            f"Expected: '{EXPECTED_MD5}'\n"
            f"Actual:   '{actual_md5}'\n"
            f"-> The file at {file_path} might be corrupted or incomplete."
        )

    seqs = []

    # Standard library fasta parsing (avoids Biopython dependency)
    with gzip.open(file_path, "rt") as f:
        seq = []
        for line in f:
            if line.startswith(">"):
                if seq:
                    joined_seq = "".join(seq)
                    # Keep only valid complete codons
                    if len(joined_seq) % 3 == 0:
                        seqs.append(joined_seq)
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            joined_seq = "".join(seq)
            if len(joined_seq) % 3 == 0:
                seqs.append(joined_seq)

    if not os.environ.get("ECOLI_FULL"):
        seqs = seqs[:ECOLI_SUBSET_SIZE]

    return seqs


@pytest.fixture
def random_seq_gen():
    """Factory fixture to generate random DNA sequences of a given length."""
    rng = np.random.default_rng()

    def _generate(length, seed=None, p=None):
        nonlocal rng
        if seed is not None:
            rng = np.random.default_rng(seed)

        bases = np.array(["A", "C", "G", "T"])
        return "".join(rng.choice(bases, size=length, p=p))

    return _generate


@pytest.fixture
def enc_default():
    """Provides a default EffectiveNumberOfCodons instance."""
    return EffectiveNumberOfCodons()


@pytest.fixture(scope="session")
def ecoli_tgcn():
    """E. coli K-12 MG1655 tRNA gene copy numbers.

    Source: https://gtrnadb.ucsc.edu/genomes/bacteria/Esch_coli_K_12_MG1655/
    """
    path = os.path.join(os.path.dirname(__file__), ".test_data", "ecoli_tgcn.csv")
    return pd.read_csv(path, comment="#")


@pytest.fixture(scope="session")
def ecoli_mrna_counts(ecoli_seqs):
    """Deterministic placeholder mRNA counts for nTE regression tests."""
    return np.ones(len(ecoli_seqs))
