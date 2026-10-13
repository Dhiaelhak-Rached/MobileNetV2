"""Phase 1: build data/splits/manifest.csv from Data/.

Columns: path, class, source_dataset, recording_id, duration_s, sample_rate,
channels, ap_split.

recording_id is the leakage-guard split unit: trailing numeric/snr tokens are
stripped so chunks, slices, and SNR variants of one session share an id.
AP_* Train/Validation/Test/UnseenDrone filename labels are captured in ap_split
and honored by the splitter.
"""
import csv
import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "data" / "splits" / "manifest.csv"
OUT_EXTRA = ROOT / "data" / "splits" / "test_unseen_extra.csv"
BAD = ROOT / "data" / "splits" / "manifest_skipped.txt"

TOKEN = re.compile(r"[_\-\s]+")


def source_dataset(stem: str) -> str:
    s = stem.lower()
    if s.startswith("unseen_") or s.startswith("new_"):
        parts = stem.split("_")
        return "_".join(parts[:-1])
    for pat, name in [
        ("dads", "DADS"), ("dronefault", "DroneFault"), ("x4", "X4"),
        ("x6", "X6"), ("membo", "Membo"), ("ap_", "AP"), ("us8k", "US8K"),
        ("esc50", "ESC50"), ("demand", "DEMAND"), ("mixed", "mixed"),
        ("synthetic", "synthetic"),
    ]:
        if s.startswith(pat):
            return name
    if re.match(r"^\d+-\d+", stem):
        return "FSDlike"
    return TOKEN.split(stem)[0]  # fallback: first token


def recording_id(source: str, stem: str) -> str:
    if stem.lower().startswith("unseen_"):
        return stem
    parts = TOKEN.split(stem)
    while parts and (parts[-1].isdigit() or parts[-1].lower().startswith("snr")):
        parts.pop()
    rid = "_".join(parts) or stem
    if source == "DADS":
        # all DADS clips would otherwise share one id; bucket by clip number
        # so buckets spread across splits while nearby clips stay together
        m = re.search(r"(\d+)", stem)
        if m:
            rid = f"DADS_b{int(m.group(1)) // 200}"
    return rid


def ap_split(stem: str) -> str:
    s = stem.lower()
    if s.startswith("unseen_"):
        return "test_unseen_extra"
    if not s.startswith("ap_"):
        return ""
    if "unseendrone" in s:
        return "test_unseen"
    for k in ("train", "validation", "test"):
        if k in s:
            return {"validation": "val"}.get(k, k)
    return ""


def main() -> None:
    rows, bad = [], []
    for cls, folder in [("drone", "Drone"), ("non_drone", "NoDrone")]:
        for f in sorted((DATA / folder).glob("*.wav")):
            try:
                info = sf.info(f)  # sniffs content: handles FLAC-in-.wav
                # prove decodability, not just header readability
                sf.read(f, frames=1600, dtype="float32", always_2d=True)
            except Exception as e:
                bad.append(f"{f}\t{e}")
                continue
            stem = f.stem
            src = source_dataset(stem)
            rows.append({
                "path": str(f),
                "class": cls,
                "source_dataset": src,
                "recording_id": recording_id(src, stem),
                "duration_s": round(info.frames / info.samplerate, 3),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "ap_split": ap_split(stem),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    BAD.write_text("\n".join(bad))

    extra_rows = [r for r in rows if r["ap_split"] == "test_unseen_extra"]
    if extra_rows:
        with OUT_EXTRA.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(extra_rows)
        print(f"wrote {len(extra_rows)} rows to {OUT_EXTRA}")

    # summary
    from collections import Counter, defaultdict
    print(f"rows: {len(rows)}  skipped: {len(bad)} (see {BAD.name})")
    hours = defaultdict(float)
    cnt = Counter()
    for r in rows:
        cnt[(r["class"], r["source_dataset"])] += 1
        hours[(r["class"], r["source_dataset"])] += r["duration_s"]
    for k in sorted(cnt):
        print(f"  {k[0]:9s} {k[1]:12s} {cnt[k]:6d} files  {hours[k]/3600:6.2f} h")
    ap = Counter(r["ap_split"] for r in rows if r["ap_split"])
    print(f"  AP prelabeled: {dict(ap)}")
    assert len(bad) == 0, f"{len(bad)} unreadable files"
    print("OK: all files decodable")


if __name__ == "__main__":
    sys.exit(main())
