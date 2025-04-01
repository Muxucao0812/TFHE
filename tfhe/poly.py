import numpy as np


def polymod(p, big_n, q=2 ** 64):
    # assume polynomial modulus is always X^N + 1
    chunks = [np.uint64(p[i : i + big_n]) for i in range(0, len(p), big_n)]
    # padd last chunk
    if len(chunks[-1]) != big_n:
        chunks[-1] = np.uint64(chunks[-1].tolist() + [0] * (big_n - len(chunks[-1])))
    sign = np.uint64(1)
    acc = np.zeros_like(chunks[0])
    for chunk in chunks:
        acc += chunk * sign
        sign *= np.uint64(-1)
    return acc


def polymul(p1, p2, q=2 ** 64):
    # using old API of numpy since it allows polymul of uint64
    return np.polymul(p1[::-1], p2[::-1])[::-1]

def naive_polymul(p1, p2, p):
    """
    Naive polynomial multiplication on ring Z[X]/(X^N + 1)
    """
    temp = np.zeros(2 * len(p1))
    for i in range(len(p1)):
        for j in range(len(p2)):
            temp[i + j] += (p1[i] * p2[j]) % p
            
    result = np.zeros(len(p1))
    for i in range(len(p1)):
        result[i] = (temp[i] - temp[i + len(p1)]) % p
    return result
    