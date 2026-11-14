from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from refinery.runner import promote

FILTERED = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("text", pa.string()),
        ("words", pa.int32()),
        ("score", pa.float32()),
        ("url_hash", pa.uint64()),
        ("text_hash", pa.uint64()),
        ("lines", pa.list_(pa.uint64())),
    ]
)
LEDGER = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("url", pa.string()),
        ("cc_lang", pa.string()),
        ("lang", pa.string()),
        ("score", pa.float32()),
        ("chars", pa.int32()),
        ("words", pa.int32()),
        ("reason", pa.string()),
        ("value", pa.float32()),
    ]
)
DROPS = pa.schema([("doc_id", pa.int64()), ("reason", pa.string()), ("value", pa.float32()), ("ref", pa.int64())])
DEDUPED = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("text", pa.string()),
        ("words", pa.int32()),
        ("lines_removed", pa.int32()),
    ]
)
CORPUS = pa.schema(
    [
        ("id", pa.int64()),
        ("url", pa.string()),
        ("crawl", pa.string()),
        ("date", pa.string()),
        ("text", pa.string()),
        ("words", pa.int32()),
        ("emails", pa.int16()),
        ("ips", pa.int16()),
        ("phones", pa.int16()),
    ]
)
class Layout:
    def __init__(self, data_dir):
        self.root = Path(data_dir)

    def crawl(self, crawl_id):
        return self.root / "crawls" / crawl_id

    def raw(self, crawl_id, index):
        return self.root / "raw" / crawl_id / f"{index:05d}.wet.gz"

    def work(self, crawl_id):
        return self.root / "work" / crawl_id

    def filtered(self, crawl_id, index):
        return self.work(crawl_id) / "filtered" / f"{index:05d}.parquet"

    def deduped(self, crawl_id, index):
        return self.work(crawl_id) / "deduped" / f"{index:05d}.parquet"

    def sigs(self, crawl_id, index):
        return self.work(crawl_id) / "sigs" / f"{index:05d}.npz"

    def lines(self, crawl_id, index):
        return self.work(crawl_id) / "lines" / f"{index:05d}.npz"

    def done(self, crawl_id, stage, index):
        return self.work(crawl_id) / "done" / stage / f"{index:05d}.json"

    def keep(self, crawl_id):
        return self.work(crawl_id) / "keep.npy"

    def clusters(self, crawl_id):
        return self.work(crawl_id) / "clusters.npz"

    def ledger(self, crawl_id, name):
        return self.root / "ledger" / crawl_id / f"{name}.parquet"

    def corpus(self, crawl_id, index):
        return self.root / "corpus" / crawl_id / f"{index:05d}.parquet"

    def report(self, crawl_id):
        return self.root / "reports" / crawl_id

    def state(self):
        return self.root / "state"



class Table:
    def __init__(self, path, schema, batch):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.schema = schema
        self.batch = batch
        self.columns = [[] for _ in schema]
        self.writer = None
        self.rows = 0

    def add(self, *values):
        for column, value in zip(self.columns, values, strict=True):
            column.append(value)
        if len(self.columns[0]) >= self.batch:
            self.flush()

    def flush(self):
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp, self.schema, compression="zstd")
        elif not self.columns[0]:
            return
        arrays = [pa.array(column, type=field.type) for column, field in zip(self.columns, self.schema, strict=True)]
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        if table.num_rows:
            self.writer.write_table(table)
        self.rows += table.num_rows
        self.columns = [[] for _ in self.schema]

    def close(self):
        self.flush()
        self.writer.close()
        promote(self.tmp, self.path)
        return self.rows
