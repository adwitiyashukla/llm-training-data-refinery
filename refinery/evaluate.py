import numpy as np
import pyarrow.parquet as pq

from refinery.bloom import Bloom
from refinery.minhash import MinHasher, candidate_probability, jaccard
from refinery.pipeline import CRAWL_STEP, SHARD_STEP, candidate_edges, load_sigs, similarity, union_find
from refinery.runner import Clock

TOP = np.iinfo(np.uint64).max
PROBES = 1_000_000
VARIANTS = (0.0, 0.6, 0.7, 0.8)


def _ranges(starts, lengths):
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    shift = starts - np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.repeat(shift, lengths) + np.arange(total)


def similarity_join(sets, threshold):
    count = len(sets)
    empty = np.zeros(0, np.int64)
    if count < 2:
        return empty, empty, np.zeros(0)
    sizes = np.array([s.size for s in sets], np.int64)
    owner = np.repeat(np.arange(count, dtype=np.int32), sizes)
    uniq, inverse, freq = np.unique(np.concatenate(sets), return_inverse=True, return_counts=True)
    rank = np.empty(uniq.size, np.int32)
    rank[np.lexsort((uniq, freq))] = np.arange(uniq.size, dtype=np.int32)
    token = rank[inverse]
    del uniq, inverse, freq, rank
    token = token[np.lexsort((token, owner))]
    starts = np.zeros(count, np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    prefix = np.maximum(0, sizes - np.ceil(threshold * sizes - 1e-9).astype(np.int64) + 1)
    inside = np.arange(token.size) - np.repeat(starts, sizes) < np.repeat(prefix, sizes)
    heads = token[inside]
    order = np.lexsort((owner[inside], heads))
    post_token, post_doc = heads[order], owner[inside][order]
    low = np.searchsorted(post_token, heads, "left")
    span = np.searchsorted(post_token, heads, "right") - low
    head_start = np.zeros(count, np.int64)
    np.cumsum(prefix[:-1], out=head_start[1:])
    busy = np.add.reduceat(span, head_start) if heads.size else np.zeros(count, np.int64)
    probes = np.lexsort((np.arange(count), sizes))
    place = np.empty(count, np.int64)
    place[probes] = np.arange(count)
    left, right, sims = [], [], []
    for x in probes.tolist():
        if prefix[x] == 0 or busy[x] <= prefix[x]:
            continue
        part = slice(head_start[x], head_start[x] + prefix[x])
        cand = post_doc[_ranges(low[part], span[part])]
        cand = np.unique(cand[place[cand] < place[x]])
        cand = cand[sizes[cand] >= threshold * sizes[x] - 1e-9]
        if cand.size == 0:
            continue
        mine = token[starts[x] : starts[x] + sizes[x]]
        values = token[_ranges(starts[cand], sizes[cand])]
        at = np.minimum(np.searchsorted(mine, values), mine.size - 1)
        overlap = np.bincount(np.repeat(np.arange(cand.size), sizes[cand]), weights=mine[at] == values,
                              minlength=cand.size)
        score = overlap / (sizes[x] + sizes[cand] - overlap)
        hit = score >= threshold - 1e-12
        left.append(np.minimum(cand[hit], x))
        right.append(np.maximum(cand[hit], x))
        sims.append(score[hit])
    if not left:
        return empty, empty, np.zeros(0)
    left, right, sims = np.concatenate(left), np.concatenate(right), np.concatenate(sims)
    order = np.lexsort((right, left))
    return left[order], right[order], sims[order]


def _shared_band(keys, a, b):
    out = np.zeros(a.size, bool)
    for start in range(0, a.size, 200_000):
        stop = start + 200_000
        out[start:stop] = (keys[a[start:stop]] == keys[b[start:stop]]).any(axis=1)
    return out


def _shingles_for(layout, crawl, hasher, doc_ids):
    wanted = {}
    for doc_id in doc_ids:
        wanted.setdefault((doc_id % CRAWL_STEP) // SHARD_STEP, []).append(doc_id)
    out = {}
    for index, members in sorted(wanted.items()):
        table = pq.read_table(layout.filtered(crawl.id, index), columns=["doc_id", "text"],
                              filters=[("doc_id", "in", members)])
        for doc_id, text in zip(table["doc_id"].to_pylist(), table["text"].to_pylist(), strict=True):
            out[doc_id] = hasher.shingles(text).astype(np.uint32)
    return out


def lsh_quality(cfg, crawl, layout, shards):
    clock = Clock()
    with np.load(layout.clusters(crawl.id)) as data:
        ids, roots, alive = data["ids"], data["roots"], data["alive"]
    data = load_sigs(layout, crawl, shards)
    sig, keys = data["sig"], data["keys"]
    position = {doc_id: i for i, doc_id in enumerate(ids.tolist())}
    hasher = MinHasher(cfg.dedup.shingle_words, cfg.dedup.bands, cfg.dedup.rows, cfg.run.seed)
    picked, sets = [], []
    for index, _ in shards[: cfg.eval.shards]:
        table = pq.read_table(layout.filtered(crawl.id, index), columns=["doc_id", "text"])
        for doc_id, text in zip(table["doc_id"].to_numpy().tolist(), table["text"].to_pylist(), strict=True):
            at = position[doc_id]
            if alive[at] and len(picked) < cfg.eval.max_docs:
                picked.append(at)
                sets.append(hasher.shingles(text).astype(np.uint32))
    picked = np.array(picked, np.int64)
    left, right, sims = similarity_join(sets, cfg.eval.floor)
    a, b = picked[left], picked[right]
    truth = sims >= cfg.eval.jaccard
    shared = _shared_band(keys, a, b)
    curve = []
    edges = np.round(np.arange(cfg.eval.floor, 1.0001, 0.05), 4)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (sims >= lo) & ((sims < hi) | (hi >= 1.0))
        if inside.any():
            curve.append({
                "low": float(lo), "high": float(hi), "pairs": int(inside.sum()),
                "measured": float(shared[inside].mean()),
                "theory": float(candidate_probability(sims[inside], cfg.dedup.bands, cfg.dedup.rows).mean()),
            })
    base_left, base_right = candidate_edges(keys, np.flatnonzero(alive))
    base_sim = similarity(sig, base_left, base_right)
    floors = sorted(set(VARIANTS) | {cfg.dedup.min_similarity})
    variant_roots = {}
    for floor in floors:
        ok = base_sim >= floor
        variant_roots[floor] = roots if floor == cfg.dedup.min_similarity else union_find(
            ids.size, base_left[ok], base_right[ok])
    slot = {g: i for i, g in enumerate(picked.tolist())}
    outside = set()
    for floor in floors:
        reps = variant_roots[floor][picked]
        outside.update(int(ids[r]) for r in reps[reps != picked].tolist() if r not in slot)
    extra = _shingles_for(layout, crawl, hasher, sorted(outside))
    should = np.unique(right[truth])
    variants = []
    for floor in floors:
        found = variant_roots[floor]
        reps = found[picked]
        gone = np.flatnonzero(reps != picked)
        scores = np.array([jaccard(sets[i], sets[slot[r]] if r in slot else extra[int(ids[r])])
                           for i, r in zip(gone.tolist(), reps[gone].tolist(), strict=True)])
        same = found[a] == found[b]
        variants.append({
            "min_similarity": floor,
            "edges": int((base_sim >= floor).sum()),
            "removed": int(gone.size),
            "removed_close": float((scores >= cfg.eval.jaccard).mean()) if gone.size else 1.0,
            "removed_loose": float((scores >= cfg.eval.floor).mean()) if gone.size else 1.0,
            "doc_recall": float((reps[should] != picked[should]).mean()) if should.size else 1.0,
            "pair_recall": float(same[truth].mean()) if truth.any() else 1.0,
        })
    chosen = next(v for v in variants if v["min_similarity"] == cfg.dedup.min_similarity)
    return {
        "sample_shards": min(cfg.eval.shards, len(shards)),
        "sample_docs": int(picked.size),
        "jaccard": cfg.eval.jaccard,
        "floor": cfg.eval.floor,
        "pairs_above_floor": int(left.size),
        "pairs_above_jaccard": int(truth.sum()),
        "docs_with_earlier_duplicate": int(should.size),
        "removed_close": chosen["removed_close"],
        "doc_recall": chosen["doc_recall"],
        "pair_recall": chosen["pair_recall"],
        "curve": curve,
        "variants": variants,
        "seconds": clock.seconds(),
    }


def bloom_quality(cfg, crawl, layout, state, shards):
    bloom = Bloom(np.load(state.path("bloom")), state.head["bloom_hashes"])
    rng = np.random.default_rng(cfg.run.seed + 1)
    probes = rng.integers(0, TOP, size=PROBES, dtype=np.uint64, endpoint=True)
    out = {
        "capacity": cfg.dedup.bloom_capacity,
        "target_fpr": cfg.dedup.bloom_fpr,
        "bits": int(bloom.size),
        "hashes": bloom.hashes,
        "fill": bloom.fill(),
        "expected_fpr": bloom.expected_fpr(),
        "probe_fpr": float(bloom.contains(probes).mean()),
        "probes": PROBES,
    }
    info = state.crawl(crawl.id)
    if info.get("lines_exact"):
        keys, hits, order = [], [], []
        for pos, (index, _) in enumerate(shards):
            with np.load(layout.lines(crawl.id, index)) as data:
                keys.append(data["keys"])
                hits.append(data["hits"])
            order.append(np.full(keys[-1].size, pos, np.int64))
        keys, hits, order = np.concatenate(keys), np.concatenate(hits), np.concatenate(order)
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        seen = order[first][inverse] < order
        out.update(
            exact_queries=int(keys.size),
            exact_new=int((~seen).sum()),
            exact_false_positives=int((hits & ~seen).sum()),
            exact_false_negatives=int((~hits & seen).sum()),
            exact_fpr=float((hits & ~seen).sum() / max(1, (~seen).sum())),
        )
    return out
