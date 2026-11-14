import numpy as np
import xxhash

from refinery.bloom import mix64
from refinery.text import simplify

BASE = np.uint64(0x100000001B3)
SHIFT = np.uint64(32)
TOP = np.iinfo(np.uint64).max


class MinHasher:
    def __init__(self, shingle_words, bands, rows, seed):
        self.width = shingle_words
        self.bands = bands
        self.rows = rows
        rng = np.random.default_rng(seed)
        self.a = rng.integers(0, TOP, size=bands * rows, dtype=np.uint64, endpoint=True) | np.uint64(1)
        self.b = rng.integers(0, TOP, size=bands * rows, dtype=np.uint64, endpoint=True)
        self.salts = rng.integers(0, TOP, size=bands, dtype=np.uint64, endpoint=True)

    def shingles(self, text):
        words = simplify(text).split() or [""]
        hashes = np.fromiter((xxhash.xxh3_64_intdigest(w.encode()) for w in words), np.uint64, len(words))
        width = min(self.width, hashes.size)
        count = hashes.size - width + 1
        keys = hashes[:count].copy()
        for j in range(1, width):
            keys = keys * BASE + hashes[j : j + count]
        return np.unique(mix64(keys) >> SHIFT)

    def signatures(self, sets):
        out = np.empty((len(sets), self.a.size), np.uint32)
        if not sets:
            return out
        values = np.concatenate(sets)
        starts = np.zeros(len(sets), np.int64)
        np.cumsum([s.size for s in sets[:-1]], out=starts[1:])
        for i in range(self.a.size):
            out[:, i] = np.minimum.reduceat((values * self.a[i] + self.b[i]) >> SHIFT, starts)
        return out

    def band_keys(self, sig):
        wide = sig.astype(np.uint64)
        out = np.empty((sig.shape[0], self.bands), np.uint64)
        for band in range(self.bands):
            key = np.full(sig.shape[0], self.salts[band], np.uint64)
            for j in range(band * self.rows, (band + 1) * self.rows):
                key = key * BASE + wide[:, j]
            out[:, band] = mix64(key)
        return out


def jaccard(a, b):
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def candidate_probability(similarity, bands, rows):
    return 1.0 - (1.0 - similarity**rows) ** bands
