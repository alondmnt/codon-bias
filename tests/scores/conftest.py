import gzip
import hashlib
import os

import numpy as np
import pytest

from codonbias.scores import EffectiveNumberOfCodons

EXPECTED_MD5 = "aaee0253df6f7d1df7df00e84d582fd4"


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
