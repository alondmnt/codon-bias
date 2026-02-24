import pytest

import os
import urllib.request
import gzip
import hashlib


EXPECTED_MD5 = "aaee0253df6f7d1df7df00e84d582fd4"


def get_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in 4K chunks to avoid loading large files entirely into memory
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


@pytest.fixture(scope="session")
def ecoli_seqs():
    """
    Downloads, parses, and caches the E. coli K-12 coding sequences.
    Scope is 'session' so it only runs once per pytest invocation.
    Includes MD5 integrity checking and self-healing cache.
    """
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz"

    # Cache the file locally in a hidden folder so it isn't re-downloaded every run
    cache_dir = os.path.join(os.path.dirname(__file__), ".test_data")
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "ecoli_cds.fna.gz")

    # 1. Validate the cache (Self-Healing)
    # If the file exists but the MD5 doesn't match, the cache is corrupt.
    if os.path.exists(file_path):
        if get_file_md5(file_path) != EXPECTED_MD5:
            os.remove(file_path)  # Remove it to force a fresh download below

    # 2. Download and verify
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
        actual_md5 = get_file_md5(file_path)

        if actual_md5 != EXPECTED_MD5:
            raise ValueError(
                f"\nMD5 mismatch for downloaded genome file!\n"
                f"Expected: '{EXPECTED_MD5}'\n"
                f"Actual:   '{actual_md5}'\n"
                f"-> If this is your first time running this, copy the 'Actual' hash above and update the EXPECTED_MD5 variable in your code."
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
