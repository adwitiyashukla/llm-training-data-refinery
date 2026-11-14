import itertools
import random

import numpy as np

from refinery.bloom import Bloom, size_for
from refinery.evaluate import similarity_join
from refinery.minhash import MinHasher, candidate_probability, jaccard
from refinery.pipeline import candidate_edges, duplicates, lookup, merge, union_find


def test_bloom_has_no_false_negatives_and_bounded_fpr():
    bloom = Bloom.empty(50_000, 0.01)
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 2**63, size=50_000, dtype=np.uint64)
    bloom.add(keys)
    assert bloom.contains(keys).all()
    probes = rng.integers(0, 2**63, size=200_000, dtype=np.uint64)
    rate = bloom.contains(probes).mean()
    assert rate < 0.02
    assert abs(rate - bloom.expected_fpr()) < 0.005


def test_bloom_size_formula():
    bits, hashes = size_for(1_000_000, 0.001)
    assert 14_300_000 < bits < 14_500_000
    assert hashes == 10


def sets_with_overlap(rng, size, similarity):
    shared = int(round(2 * size * similarity / (1 + similarity)))
    pool = rng.choice(2**32, size=2 * size, replace=False).astype(np.uint64)
    a = pool[:size]
    b = np.concatenate([a[:shared], pool[size : 2 * size - shared]])
    return np.sort(a), np.sort(b)


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(2)
    hasher = MinHasher(5, 14, 8, seed=3)
    errors = []
    for target in (0.2, 0.5, 0.8, 0.95):
        for _ in range(30):
            a, b = sets_with_overlap(rng, 400, target)
            sig = hasher.signatures([a, b])
            errors.append((sig[0] == sig[1]).mean() - jaccard(a, b))
    errors = np.array(errors)
    assert abs(errors.mean()) < 0.02
    assert np.abs(errors).max() < 0.2


def test_lsh_candidate_rate_follows_theory():
    rng = np.random.default_rng(4)
    hasher = MinHasher(5, 14, 8, seed=5)
    for target in (0.5, 0.7, 0.85):
        hits = []
        for _ in range(300):
            a, b = sets_with_overlap(rng, 300, target)
            keys = hasher.band_keys(hasher.signatures([a, b]))
            hits.append((keys[0] == keys[1]).any())
        expected = candidate_probability(target, 14, 8)
        assert abs(np.mean(hits) - expected) < 0.1


def test_shingles_are_order_sensitive_and_stable():
    hasher = MinHasher(5, 14, 8, seed=1)
    text = "the quick brown fox jumps over the lazy dog near the river bank"
    assert np.array_equal(hasher.shingles(text), hasher.shingles(text.upper()))
    assert not np.array_equal(hasher.shingles(text), hasher.shingles(" ".join(reversed(text.split()))))
    assert hasher.shingles("tiny doc").size == 1


def components(size, left, right):
    graph = {i: set() for i in range(size)}
    for a, b in zip(left, right, strict=True):
        graph[a].add(b)
        graph[b].add(a)
    roots = [-1] * size
    for start in range(size):
        if roots[start] >= 0:
            continue
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            for nxt in graph[node] - seen:
                seen.add(nxt)
                stack.append(nxt)
        for node in seen:
            roots[node] = min(seen)
    return roots


def test_union_find_matches_graph_search():
    rng = random.Random(6)
    for _ in range(50):
        size = rng.randint(1, 80)
        edges = rng.randint(0, 120)
        left = np.array([rng.randrange(size) for _ in range(edges)], np.int64)
        right = np.array([rng.randrange(size) for _ in range(edges)], np.int64)
        assert union_find(size, left, right).tolist() == components(size, left.tolist(), right.tolist())


def test_candidate_edges_link_docs_sharing_any_band():
    keys = np.array([[1, 2], [3, 2], [4, 5], [6, 7], [4, 9]], np.uint64)
    left, right = candidate_edges(keys, np.arange(5))
    assert sorted(zip(left.tolist(), right.tolist(), strict=True)) == [(0, 1), (2, 4)]
    left, right = candidate_edges(keys, np.array([1, 2, 3]))
    assert left.size == 0


def test_similarity_join_matches_brute_force():
    rng = np.random.default_rng(7)
    sets = []
    base = rng.choice(10_000, size=60, replace=False).astype(np.uint32)
    for _ in range(40):
        keep = base[rng.random(base.size) < rng.uniform(0.3, 1.0)]
        extra = rng.choice(10_000, size=rng.integers(0, 30), replace=False).astype(np.uint32)
        sets.append(np.unique(np.concatenate([keep, extra])))
    for threshold in (0.3, 0.5, 0.75):
        left, right, sims = similarity_join(sets, threshold)
        found = {(a, b): s for a, b, s in zip(left.tolist(), right.tolist(), sims.tolist(), strict=True)}
        expected = {}
        for a, b in itertools.combinations(range(len(sets)), 2):
            value = jaccard(sets[a], sets[b])
            if value >= threshold:
                expected[(a, b)] = value
        assert found.keys() == expected.keys()
        assert all(abs(found[k] - expected[k]) < 1e-12 for k in expected)


def test_lookup_merge_and_duplicates():
    keys, refs = merge(np.zeros(0, np.uint64), np.zeros(0, np.int64),
                       np.array([5, 3, 5], np.uint64), np.array([10, 11, 12], np.int64))
    assert keys.tolist() == [3, 5] and refs.tolist() == [11, 10]
    assert lookup(keys, refs, np.array([5, 4], np.uint64)).tolist() == [10, -1]
    dup, ref = duplicates(np.array([7, 3, 7, 9], np.uint64), np.array([20, 21, 22, 23], np.int64), keys, refs)
    assert dup.tolist() == [False, True, True, False]
    assert ref.tolist() == [20, 11, 20, 23]
