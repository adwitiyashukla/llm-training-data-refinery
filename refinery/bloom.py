import math

import numpy as np

C1 = np.uint64(0xBF58476D1CE4E5B9)
C2 = np.uint64(0x94D049BB133111EB)
CHUNK = 200_000


def mix64(values):
    values = values ^ (values >> np.uint64(30))
    values = values * C1
    values = values ^ (values >> np.uint64(27))
    values = values * C2
    return values ^ (values >> np.uint64(31))


def size_for(capacity, fpr):
    bits = math.ceil(-capacity * math.log(fpr) / math.log(2) ** 2)
    bits = (bits + 63) // 64 * 64
    hashes = max(1, round(bits / capacity * math.log(2)))
    return bits, hashes


class Bloom:
    def __init__(self, bits, hashes):
        self.bits = bits
        self.hashes = hashes
        self.size = np.uint64(bits.size * 8)
        self.steps = np.arange(hashes, dtype=np.uint64)

    @classmethod
    def empty(cls, capacity, fpr):
        bits, hashes = size_for(capacity, fpr)
        return cls(np.zeros(bits // 8, np.uint8), hashes)

    def _positions(self, keys):
        second = mix64(keys) | np.uint64(1)
        return (keys[:, None] + self.steps[None, :] * second[:, None]) % self.size

    def contains(self, keys):
        keys = np.asarray(keys, np.uint64)
        out = np.empty(keys.size, bool)
        for start in range(0, keys.size, CHUNK):
            pos = self._positions(keys[start : start + CHUNK])
            hit = (self.bits[(pos >> np.uint64(3)).astype(np.int64)] >> (pos & np.uint64(7)).astype(np.uint8)) & 1
            out[start : start + CHUNK] = hit.all(axis=1)
        return out

    def add(self, keys):
        keys = np.asarray(keys, np.uint64)
        for start in range(0, keys.size, CHUNK):
            pos = self._positions(keys[start : start + CHUNK]).ravel()
            masks = np.left_shift(1, (pos & np.uint64(7)).astype(np.uint8)).astype(np.uint8)
            np.bitwise_or.at(self.bits, (pos >> np.uint64(3)).astype(np.int64), masks)

    def fill(self):
        return float(np.bitwise_count(self.bits).sum()) / float(self.size)

    def expected_fpr(self):
        return self.fill() ** self.hashes
