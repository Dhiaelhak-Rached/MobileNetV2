"""Phase 2: recording-level train/val/test split.

Rules (leakage guards):
- Split unit is recording_id, never an individual file/chunk.
- AP_* files keep their filename-encoded labels (train/val/test/test_unseen);
  test_unseen folds into test.
- Everything else: 70/15/15 per (class, source_dataset) group, duration-weighted
  greedy assignment over shuffled recording_ids.

Outputs data/splits/{train,val,test}.csv and asserts zero recording_id overlap.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "splits" / "manifest.csv"
OUTDIR = ROOT / "data" / "splits"
SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))

    # group rows by recording_id
    groups = defaultdict(list)
    for r in rows:
        groups[r["recording_id"]].append(r)

    assignment = {}  # recording_id -> split

    # 1. honor AP prelabels
    for rid, rs in groups.items():
        ap = {r["ap_split"] for r in rs if r["ap_split"]}
        if ap:
            assert len(ap) == 1, f"{rid}: conflicting AP labels {ap}"
            label = ap.pop()
            assignment[rid] = "test" if label == "test_unseen" else label

    # 2. stratified split of the rest, per (class, source)
    buckets = defaultdict(list)  # (class, source) -> [(rid, hours)]
    for rid, rs in groups.items():
        if rid in assignment:
            continue
        key = (rs[0]["class"], rs[0]["source_dataset"])
        hours = sum(float(r["duration_s"]) for r in rs)
        buckets[key].append((rid, hours))

    rng = random.Random(SEED)
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        items.sort(key=lambda x: -x[1])  # big recordings first
        total = sum(h for _, h in items)
        fill = {"train": 0.0, "val": 0.0, "test": 0.0}
        for rid, h in items:
            # split furthest below its target share
            split = min(RATIOS, key=lambda s: fill[s] / (RATIOS[s] * total) if total else 0)
            assignment[rid] = split
            fill[split] += h

    # 3. write splits + leakage check
    seen = defaultdict(set)
    out_rows = defaultdict(list)
    for r in rows:
        split = assignment[r["recording_id"]]
        seen[split].add(r["recording_id"])
        out_rows[split].append(r)

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = seen[a] & seen[b]
        assert not overlap, f"LEAKAGE: {len(overlap)} recording_ids in both {a} and {b}"

    for split, rs in out_rows.items():
        with (OUTDIR / f"{split}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rs)

    # 4. summary
    print(f"seed={SEED} ratios={RATIOS}  (AP prelabels honored)")
    for split in ("train", "val", "test"):
        rs = out_rows[split]
        hrs = defaultdict(float)
        for r in rs:
            hrs[r["class"]] += float(r["duration_s"]) / 3600
        tot = sum(hrs.values())
        print(f"{split:5s}: {len(rs):5d} files, {tot:6.2f} h "
              f"(drone {hrs['drone']:.2f} h / non_drone {hrs['non_drone']:.2f} h)")
    print("OK: zero recording_id overlap between splits")


if __name__ == "__main__":
    main()
