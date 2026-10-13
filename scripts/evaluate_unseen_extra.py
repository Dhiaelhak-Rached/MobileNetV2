"""Comprehensive generalization evaluation for unseen drone models/families.

Evaluates data/splits/test_unseen_extra.csv against the frozen model checkpoint:
- Reports P(drone) distribution (mean, min, max, std)
- Reports recall at tau=0.50 and tau=0.70 (frozen default)
- Identifies files below thresholds
- Categorizes models into Reliable (>=80%), Moderately Reliable (50-79%), Failed (<50%)
- Breakdown by propeller size / drone type
- Breakdown by distance / gain
- Breakdown by real vs synthetic audio
Saves results to results/unseen_extra_eval.json and prints summary tables.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(r"E:\DroneDetectionModel")
sys.path.insert(0, str(ROOT))

from src.audio.preprocessing import AudioConfig, Preprocessor
from src.dataset.dataset import DroneWaveDataset
from src.evaluation.common import load_model
from src.training.train import to_spec


# Drone model categorization metadata
MODEL_CATEGORIES = {
    # 1. Mini / Micro multirotors (small props <4 inches, ultra-high RPM)
    "DJI_Tello": {"prop_type": "Micro (<4 in)", "category": "Micro Multirotor", "speed": "Low/Hover"},
    "DJI_Avata": {"prop_type": "Ducted 5-blade (2.9 in)", "category": "CineWhoop / Micro", "speed": "Variable"},
    "Aerobatic_Freestyle": {"prop_type": "5-inch 3-blade", "category": "Acrobatic / Freestyle", "speed": "Aggressive"},

    # 2. Racing / Tactical strike FPV (high RPM screams, sharp throttle transitions)
    "DJI_FPV": {"prop_type": "5.3-inch 3-blade", "category": "Racing FPV", "speed": "High"},
    "FPV_Kamikaze": {"prop_type": "7-10 inch multirotor", "category": "Tactical Strike FPV", "speed": "High / Terminal"},

    # 3. Standard / Commercial multirotor (large props 9-10 inches, lower acoustic pitch)
    "DJI_FlameWheel_F450": {"prop_type": "9-10 inch 2-blade", "category": "Commercial Multirotor", "speed": "Low/Hover"},

    # 4. Fixed-wing / Long-range UAV / Internal combustion engines (distinct engine harmonics)
    "Shahed_136_Combat": {"prop_type": "MD-550 50hp pusher", "category": "Delta-wing Loitering Munition", "speed": "Cruise / Attack"},
    "Shahed_136_Transit": {"prop_type": "MD-550 50hp pusher", "category": "Delta-wing Loitering Munition", "speed": "High-altitude Transit"},
    "Geran_2": {"prop_type": "MD-550 50hp pusher", "category": "Delta-wing Loitering Munition", "speed": "Transit"},
    "Geran_3": {"prop_type": "Small turbojet / turbine", "category": "Jet Delta-wing UAV", "speed": "Very High"},
    "Gerbera": {"prop_type": "Small pusher prop", "category": "Decoy / Recon UAV", "speed": "Cruise"},
    "Limbach_L550E": {"prop_type": "Limbach L550E 50hp engine", "category": "UAV Engine Ground Bench", "speed": "Bench / Varied RPM"},

    # 5. Synthetic aerodynamic baselines (physics-grounded aeroacoustic harmonics)
    "Synthetic_BPF_Hover": {"prop_type": "Aeroacoustic 180Hz BPF", "category": "Physics Synthesis", "speed": "Hover"},
    "Synthetic_BPF_HighSpeed": {"prop_type": "Aeroacoustic Doppler 300Hz BPF", "category": "Physics Synthesis", "speed": "20m/s Fly-by"},
}


def score_unseen(model, pp, rows, device="cuda"):
    """Score all rows with the frozen model.
    Returns list of dicts with full sample info, prediction score, and threshold flags.
    """
    results = []
    model.eval()
    with torch.no_grad():
        for r in rows:
            wav_path = r["path"]
            # Load and preprocess through official Preprocessor
            w = pp.load(wav_path)
            # Standard center-crop 1-second segment (identical to test set evaluation)
            seg = pp.fix_length(w, offset=None).unsqueeze(0).to(device)
            spec = to_spec(pp, seg, device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(spec)
            p_drone = float(torch.softmax(logits.float(), dim=1)[0, 1].cpu().item())

            # Also compute multi-window / full-audio statistics if audio > 1s
            full_w = w
            step = pp.cfg.n_samples // 2  # 50% overlap
            n_samples = pp.cfg.n_samples
            if full_w.numel() > n_samples:
                windows = []
                for start in range(0, full_w.numel() - n_samples + 1, step):
                    win = full_w[start : start + n_samples].unsqueeze(0).to(device)
                    win_spec = to_spec(pp, win, device)
                    with torch.amp.autocast("cuda", enabled=device == "cuda"):
                        w_logits = model(win_spec)
                    w_prob = float(torch.softmax(w_logits.float(), dim=1)[0, 1].cpu().item())
                    windows.append(w_prob)
                p_mean_win = float(np.mean(windows))
                p_max_win = float(np.max(windows))
            else:
                p_mean_win = p_drone
                p_max_win = p_drone

            results.append({
                "path": str(wav_path),
                "filename": Path(wav_path).name,
                "model": r["source_dataset"].replace("UNSEEN_", ""),
                "source_dataset": r["source_dataset"],
                "duration_s": float(r["duration_s"]),
                "p_drone": p_drone,
                "p_drone_mean_windows": p_mean_win,
                "p_drone_max_windows": p_max_win,
                "det_tau_050": p_drone >= 0.50,
                "det_tau_070": p_drone >= 0.70,
            })
    return results


def analyze_unseen(scores, manifest_meta):
    """Compute aggregate and per-model metrics."""
    by_model = defaultdict(list)
    for s in scores:
        by_model[s["model"]].append(s)

    per_model_results = {}
    reliable_models = []
    moderate_models = []
    failed_models = []

    for model_name, items in sorted(by_model.items()):
        probs = [it["p_drone"] for it in items]
        durations = [it["duration_s"] for it in items]
        n_files = len(items)
        tot_dur = sum(durations)

        rec_50 = sum(1 for p in probs if p >= 0.50) / n_files
        rec_70 = sum(1 for p in probs if p >= 0.70) / n_files

        below_50 = [it["filename"] for it in items if it["p_drone"] < 0.50]
        below_70 = [it["filename"] for it in items if it["p_drone"] < 0.70]

        is_synth = manifest_meta.get(items[0]["filename"], {}).get("synthetic", False)
        source_desc = manifest_meta.get(items[0]["filename"], {}).get("source", "Unknown")

        model_stat = {
            "model": model_name,
            "category_info": MODEL_CATEGORIES.get(model_name, {}),
            "synthetic": is_synth,
            "source_sample": source_desc,
            "n_files": n_files,
            "total_duration_s": round(tot_dur, 2),
            "p_drone_mean": float(np.mean(probs)),
            "p_drone_std": float(np.std(probs)),
            "p_drone_min": float(np.min(probs)),
            "p_drone_median": float(np.median(probs)),
            "p_drone_max": float(np.max(probs)),
            "recall_tau_050": round(rec_50, 4),
            "recall_tau_070": round(rec_70, 4),
            "n_below_tau_050": len(below_50),
            "n_below_tau_070": len(below_70),
            "files_below_tau_050": below_50,
            "files_below_tau_070": below_70,
        }
        per_model_results[model_name] = model_stat

        # Grouping
        if rec_70 >= 0.80:
            reliable_models.append(model_name)
        elif rec_70 >= 0.50:
            moderate_models.append(model_name)
        else:
            failed_models.append(model_name)

    # Breakdown by category
    by_category = defaultdict(list)
    for s in scores:
        cat = MODEL_CATEGORIES.get(s["model"], {}).get("category", "Other")
        by_category[cat].append(s)

    cat_results = {}
    for cat, items in sorted(by_category.items()):
        probs = [it["p_drone"] for it in items]
        cat_results[cat] = {
            "n_files": len(items),
            "total_duration_s": round(sum(it["duration_s"] for it in items), 2),
            "p_drone_mean": float(np.mean(probs)),
            "recall_tau_050": round(sum(1 for p in probs if p >= 0.50) / len(items), 4),
            "recall_tau_070": round(sum(1 for p in probs if p >= 0.70) / len(items), 4),
        }

    # Breakdown by Real vs Synthetic
    real_items = [s for s in scores if not manifest_meta.get(s["filename"], {}).get("synthetic", False)]
    synth_items = [s for s in scores if manifest_meta.get(s["filename"], {}).get("synthetic", False)]

    real_probs = [s["p_drone"] for s in real_items]
    synth_probs = [s["p_drone"] for s in synth_items]

    real_vs_synth = {
        "real": {
            "n_files": len(real_items),
            "total_duration_s": round(sum(it["duration_s"] for it in real_items), 2),
            "p_drone_mean": float(np.mean(real_probs)) if real_probs else 0.0,
            "recall_tau_050": round(sum(1 for p in real_probs if p >= 0.50) / len(real_items), 4) if real_items else 0.0,
            "recall_tau_070": round(sum(1 for p in real_probs if p >= 0.70) / len(real_items), 4) if real_items else 0.0,
        },
        "synthetic": {
            "n_files": len(synth_items),
            "total_duration_s": round(sum(it["duration_s"] for it in synth_items), 2),
            "p_drone_mean": float(np.mean(synth_probs)) if synth_probs else 0.0,
            "recall_tau_050": round(sum(1 for p in synth_probs if p >= 0.50) / len(synth_items), 4) if synth_items else 0.0,
            "recall_tau_070": round(sum(1 for p in synth_probs if p >= 0.70) / len(synth_items), 4) if synth_items else 0.0,
        }
    }

    # Global unseen summary
    all_probs = [s["p_drone"] for s in scores]
    summary = {
        "total_files": len(scores),
        "total_duration_s": round(sum(s["duration_s"] for s in scores), 2),
        "total_models": len(by_model),
        "p_drone_mean": float(np.mean(all_probs)),
        "p_drone_median": float(np.median(all_probs)),
        "recall_tau_050": round(sum(1 for p in all_probs if p >= 0.50) / len(scores), 4),
        "recall_tau_070": round(sum(1 for p in all_probs if p >= 0.70) / len(scores), 4),
        "reliable_models": reliable_models,
        "moderately_reliable_models": moderate_models,
        "failed_models": failed_models,
        "per_model": per_model_results,
        "by_category": cat_results,
        "real_vs_synthetic": real_vs_synth,
        "individual_predictions": scores,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(ROOT / "artifacts" / "model" / "mobilenetv2_drone_best.pth"))
    parser.add_argument("--split", default=str(ROOT / "data" / "splits" / "test_unseen_extra.csv"))
    parser.add_argument("--meta", default=str(ROOT / "results" / "unseen_sources_manifest.json"))
    parser.add_argument("--out", default=str(ROOT / "results" / "unseen_extra_eval.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading frozen model from {args.ckpt} on {device}...")
    model, pp, _ = load_model(args.ckpt, device=device)

    print(f"Reading unseen split from {args.split}...")
    with open(args.split, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} audio rows.")

    manifest_meta = {}
    if Path(args.meta).exists():
        manifest_meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    print("Running inference across unseen drone models...")
    scores = score_unseen(model, pp, rows, device=device)

    print("Analyzing generalization performance...")
    eval_summary = analyze_unseen(scores, manifest_meta)

    # Save to json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation results to {out_path}\n")

    # Print Summary Table
    print("=" * 110)
    print(f"{'Unseen Drone Model/Family':30s} {'Type':10s} {'Files':>5} {'Dur(s)':>7} {'P(mean)':>8} {'P(min)':>8} {'P(max)':>8} {'Rec@0.5':>8} {'Rec@0.7':>8} {'Status':>10}")
    print("=" * 110)
    for model_name, stat in sorted(eval_summary["per_model"].items()):
        typ = "Synthetic" if stat["synthetic"] else "Real"
        status = "Reliable" if stat["recall_tau_070"] >= 0.8 else ("Moderate" if stat["recall_tau_070"] >= 0.5 else "FAIL")
        print(f"{model_name:30s} {typ:10s} {stat['n_files']:5d} {stat['total_duration_s']:7.1f} {stat['p_drone_mean']:8.4f} {stat['p_drone_min']:8.4f} {stat['p_drone_max']:8.4f} {stat['recall_tau_050']:8.2%} {stat['recall_tau_070']:8.2%} {status:>10s}")
    print("-" * 110)
    g = eval_summary
    print(f"{'OVERALL UNSEEN EVALUATION':30s} {'All':10s} {g['total_files']:5d} {g['total_duration_s']:7.1f} {g['p_drone_mean']:8.4f} {'-':>8} {'-':>8} {g['recall_tau_050']:8.2%} {g['recall_tau_070']:8.2%} {'-':>10}")
    print("=" * 110)

    print("\n=== Model Reliability Grouping (at default frozen tau = 0.70) ===")
    print(f"Reliable (Recall >= 80%):        {', '.join(eval_summary['reliable_models']) or 'None'}")
    print(f"Moderately Reliable (50-79%):     {', '.join(eval_summary['moderately_reliable_models']) or 'None'}")
    print(f"Failed Models (Recall < 50%):     {', '.join(eval_summary['failed_models']) or 'None'}")

    print("\n=== Category Breakdown ===")
    print(f"{'Category':32s} {'Files':>5} {'Dur(s)':>8} {'P(mean)':>8} {'Rec@0.5':>8} {'Rec@0.7':>8}")
    print("-" * 75)
    for cat, stat in sorted(eval_summary["by_category"].items()):
        print(f"{cat:32s} {stat['n_files']:5d} {stat['total_duration_s']:8.1f} {stat['p_drone_mean']:8.4f} {stat['recall_tau_050']:8.2%} {stat['recall_tau_070']:8.2%}")

    print("\n=== Real vs Synthetic Audio ===")
    for k, stat in eval_summary["real_vs_synthetic"].items():
        print(f"  {k.capitalize():10s}: {stat['n_files']:3d} files ({stat['total_duration_s']:6.1f}s), P(mean)={stat['p_drone_mean']:.4f}, Recall@0.50={stat['recall_tau_050']:.2%}, Recall@0.70={stat['recall_tau_070']:.2%}")

if __name__ == "__main__":
    main()
