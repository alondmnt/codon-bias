# cython: language_level=3
import numpy as np
cimport cython

# Static C-arrays: Baked into the binary
cdef int CODON_TO_AA[64]
CODON_TO_AA[:] = [
    8, 11, 8, 11, 16, 16, 16, 16, 14, 15, 14, 15, 7, 7, 10, 7,
    13, 6, 13, 6, 12, 12, 12, 12, 14, 14, 14, 14, 9, 9, 9, 9,
    3, 2, 3, 2, 0, 0, 0, 0, 5, 5, 5, 5, 17, 17, 17, 17,
    -1, 19, -1, 19, 15, 15, 15, 15, -1, 1, 18, 1, 9, 4, 9, 4
]

cdef int AA_DEG[20]
AA_DEG[:] = [4, 2, 2, 2, 2, 4, 2, 3, 2, 6, 1, 2, 4, 2, 6, 6, 4, 4, 1, 2]

cdef double DEG_COUNTS[5]
DEG_COUNTS[:] = [2.0, 9.0, 1.0, 5.0, 3.0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int char_to_int(unsigned char c) nogil:
    if c == 65 or c == 97: return 0  # A/a
    if c == 67 or c == 99: return 1  # C/c
    if c == 71 or c == 103: return 2  # G/g
    if c == 84 or c == 116 or c == 85 or c == 117: return 3  # T/t/U/u
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_enc_core_cython(const unsigned char[:] seq_view, bint bg_correction):
    cdef int n_len = seq_view.shape[0]
    if n_len == 0: return float('nan')

    # Get a raw C pointer to the sequence data
    cdef const unsigned char * seq_bytes = &seq_view[0]

    cdef double bnc_counts[4]
    cdef double cod_counts[64]
    cdef double BCC[64]
    cdef double BNC[4]
    cdef double F_aa[20]
    cdef double N_aa[20]
    cdef double F_deg_sum[5]
    cdef double N_deg_sum[5]
    cdef double F_deg[5]

    cdef Py_ssize_t i  # Use Py_ssize_t for indices
    cdef int b1, b2, b3, idx, a, deg, deg_idx
    cdef double bnc_sum, sum_bcc, n_sum, chi2_a, P_idx, diff, ENC = 0.0

    # nogil ensures no Python interaction can happen inside this block
    with nogil:
        # Initialize
        for i in range(64): cod_counts[i] = 0.0
        for i in range(5):
            F_deg_sum[i] = 0.0
            N_deg_sum[i] = 0.0

        # 1. Byte-loop counting (RAW POINTER ACCESS)
        for i in range(0, n_len - 2, 3):
            b1 = char_to_int(seq_bytes[i])
            b2 = char_to_int(seq_bytes[i + 1])
            b3 = char_to_int(seq_bytes[i + 2])
            if b1 >= 0 and b2 >= 0 and b3 >= 0:
                idx = b1 * 16 + b2 * 4 + b3
                if CODON_TO_AA[idx] != -1:
                    cod_counts[idx] += 1.0

        # 2. Background Composition
        if bg_correction:
            for i in range(4): bnc_counts[i] = 1.0
            for i in range(n_len):
                idx = char_to_int(seq_bytes[i])
                if idx >= 0: bnc_counts[idx] += 1.0

            bnc_sum = bnc_counts[0] + bnc_counts[1] + bnc_counts[2] + bnc_counts[3]
            for i in range(4): BNC[i] = bnc_counts[i] / bnc_sum

            for idx in range(64):
                a = CODON_TO_AA[idx]
                if a != -1:
                    BCC[idx] = BNC[idx // 16] * BNC[(idx // 4) % 4] * BNC[idx % 4]

            for a in range(20):
                sum_bcc = 0.0
                for idx in range(64):
                    if CODON_TO_AA[idx] == a: sum_bcc += BCC[idx]
                if sum_bcc > 0.0:
                    for idx in range(64):
                        if CODON_TO_AA[idx] == a: BCC[idx] /= sum_bcc
        else:
            for idx in range(64):
                a = CODON_TO_AA[idx]
                if a != -1: BCC[idx] = 1.0 / AA_DEG[a]

        # 3. ENC Core Math
        for a in range(20):
            n_sum = 0.0
            for idx in range(64):
                if CODON_TO_AA[idx] == a: n_sum += (cod_counts[idx] + 1.0)
            N_aa[a] = n_sum

            chi2_a = 0.0
            for idx in range(64):
                if CODON_TO_AA[idx] == a:
                    P_idx = (cod_counts[idx] + 1.0) / n_sum
                    diff = P_idx - BCC[idx]
                    chi2_a += (diff * diff) / BCC[idx]
            F_aa[a] = (chi2_a + 1.0) / AA_DEG[a]

        # 4. Aggregation by Degeneracy
        for a in range(20):
            deg = AA_DEG[a]
            deg_idx = 0 if deg == 1 else 1 if deg == 2 else 2 if deg == 3 else 3 if deg == 4 else 4
            F_deg_sum[deg_idx] += F_aa[a] * N_aa[a]
            N_deg_sum[deg_idx] += N_aa[a]

        for i in range(5):
            if N_deg_sum[i] > 0.0:
                F_deg[i] = F_deg_sum[i] / N_deg_sum[i]
            else:
                F_deg[i] = -1.0  # Use -1.0 for NaN in nogil

        # Fix missing deg=3
        if F_deg[2] < 0:
            F_deg[2] = 0.5 * (F_deg[1] + F_deg[3])

        # 5. Final Sum
        for i in range(5):
            if F_deg[i] > 0:
                ENC += DEG_COUNTS[i] / F_deg[i]

    return 61.0 if ENC > 61.0 else ENC