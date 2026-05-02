"""Benchmark: compare mutation search speed across optimization configurations.

Modes:
  original      — per-string features, batch predict, no cascade, no CUDA
  vectorized    — vectorized batch features, batch predict, no cascade, no CUDA
  cascade       — per-string features, batch predict, cascade filtering, no CUDA
  cuda          — per-string features, batch predict with CUDA, no cascade
  full          — vectorized + cascade + CUDA (current pipeline)

Usage:
  conda run -n aptamer-predictor python benchmark/benchmark.py [options]
  conda run -n aptamer-predictor python benchmark/benchmark.py --sites 8 --timeout 120
  conda run -n aptamer-predictor python benchmark/benchmark.py -o benchmark/results.csv
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Add project root so we can import aptamer_predictor
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from aptamer_predictor.features import (
    MER_K_MAP,
    build_feature_matrix,
    build_feature_vector_fast,
    molecular_descriptors,
    rna_to_dna,
)
from aptamer_predictor.predictor import EnsemblePredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
DEFAULT_SEQUENCE = "GACGACTAAAAAAAAAAAAAAAAAGTCGTC"
DEFAULT_SITE_COUNTS = [7, 8, 9, 10, 11, 12, 13]
CHUNK_SIZE = 65536

MODE_NAMES = ["original", "vectorized", "cascade", "cuda", "full"]


@dataclass
class BenchResult:
    mode: str
    n_sites: int
    n_candidates: int
    elapsed: Optional[float] = None
    n_positives: int = 0


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _progress_line(done: int, total: int, start_time: float) -> str:
    elapsed = time.perf_counter() - start_time
    pct = done / total * 100 if total else 0
    speed = done / elapsed if elapsed > 0.1 else 0
    if speed >= 1_000_000:
        speed_s = f"{speed / 1_000_000:.1f}M/s"
    elif speed >= 1_000:
        speed_s = f"{speed / 1_000:.1f}K/s"
    else:
        speed_s = f"{speed:.0f}/s"
    return f"\r  {pct:5.1f}% | {done:>10,} / {total:,} | {speed_s}"


# ---------------------------------------------------------------------------
# Mutant enumeration
# ---------------------------------------------------------------------------

def _enumerate_string_chunks(seq: str, sites: list[int], cancel: threading.Event,
                             chunk_size: int = CHUNK_SIZE):
    """Yield chunks of mutant strings (for non-vectorized modes)."""
    bases = ["A", "T", "G", "C"]
    seq_list = list(seq)
    chunk: list[str] = []
    for combo in product(bases, repeat=len(sites)):
        if cancel.is_set():
            return
        mutant = seq_list.copy()
        for pos, base in zip(sites, combo):
            mutant[pos] = base
        chunk.append("".join(mutant))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _enumerate_numpy_chunks(seq: str, sites: list[int], cancel: threading.Event,
                            chunk_size: int = CHUNK_SIZE):
    """Yield chunks of mutant byte arrays (for vectorized modes)."""
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    bases_bytes = np.frombuffer(b"ATGC", dtype=np.uint8)
    sites_arr = np.array(sites, dtype=np.intp)
    total = 4 ** len(sites)

    for start in range(0, total, chunk_size):
        if cancel.is_set():
            break
        stop = min(start + chunk_size, total)
        batch_len = stop - start
        digits = np.empty((batch_len, len(sites)), dtype=np.int8)
        values = np.arange(start, stop, dtype=np.int64)
        for pos in range(len(sites) - 1, -1, -1):
            digits[:, pos] = values % 4
            values //= 4
        mutant_bytes = np.tile(seq_bytes, (batch_len, 1))
        mutant_bytes[:, sites_arr] = bases_bytes[digits]
        yield mutant_bytes


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate_models(models, seq, sites, desc, cancel):
    """Evaluate 64 sample mutants, return models sorted by selectivity."""
    bases = ["A", "T", "G", "C"]
    seq_list = list(seq)
    calib_seqs = []
    for combo in product(bases, repeat=len(sites)):
        if len(calib_seqs) >= 64:
            break
        mutant = seq_list.copy()
        for pos, base in zip(sites, combo):
            mutant[pos] = base
        calib_seqs.append("".join(mutant))

    scored = []
    for original_idx, (model, mer, fname) in enumerate(models):
        if cancel.is_set():
            break
        if mer is None or mer not in MER_K_MAP:
            continue
        k_list = MER_K_MAP[mer]
        X = build_feature_matrix(calib_seqs, desc, k_list)
        preds = model.predict(X)
        n_pos = int((preds >= 0.5).sum())
        scored.append((n_pos, original_idx, model, mer, fname, k_list))

    scored.sort(key=lambda x: x[0])
    return [(o, m, mer, fn, kl) for _, o, m, mer, fn, kl in scored]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_models(models):
    return [
        (i, m, mer, fn)
        for i, (m, mer, fn) in enumerate(models)
        if mer is not None and mer in MER_K_MAP
    ]


def _predict_cpu(model, X):
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            probs = model.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int), probs
    except ImportError:
        pass
    probs = model.predict_proba(X)[:, 1]
    return (probs >= 0.5).astype(int), probs


def _build_feat_matrix_from_strings(strings, desc, k_list):
    """Build feature matrix from a list of strings (per-string k-mer, batch stack)."""
    feats = [build_feature_vector_fast(s, desc, k_list) for s in strings]
    return np.vstack(feats)


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------

def benchmark_original(models, seq, desc, sites, cancel):
    """Per-string features, batch predict, no cascade, no CUDA."""
    active = _get_active_models(models)
    total = 4 ** len(sites)
    positives = 0
    processed = 0
    start = time.perf_counter()

    for chunk in _enumerate_string_chunks(seq, sites, cancel):
        if cancel.is_set():
            break
        B = len(chunk)
        all_positive = np.ones(B, dtype=bool)
        for _, model, mer, _ in active:
            X = _build_feat_matrix_from_strings(chunk, desc, MER_K_MAP[mer])
            preds, _ = _predict_cpu(model, X)
            all_positive &= (preds >= 0.5)
        positives += int(all_positive.sum())
        processed += B
        sys.stderr.write(_progress_line(processed, total, start))
        sys.stderr.flush()

    elapsed = time.perf_counter() - start
    return elapsed, positives


def benchmark_vectorized(models, seq, desc, sites, cancel):
    """Vectorized batch features, batch predict, no cascade, no CUDA."""
    active = _get_active_models(models)
    total = 4 ** len(sites)
    positives = 0
    processed = 0
    start = time.perf_counter()

    for chunk in _enumerate_numpy_chunks(seq, sites, cancel):
        if cancel.is_set():
            break
        B = chunk.shape[0]
        positive_counts = np.zeros(B, dtype=np.int32)
        for _, model, mer, _ in active:
            X = build_feature_matrix(chunk, desc, MER_K_MAP[mer])
            preds, _ = _predict_cpu(model, X)
            positive_counts += (preds >= 0.5).astype(np.int32)
        positives += int((positive_counts == len(active)).sum())
        processed += B
        sys.stderr.write(_progress_line(processed, total, start))
        sys.stderr.flush()

    elapsed = time.perf_counter() - start
    return elapsed, positives


def benchmark_cascade(models, seq, desc, sites, cancel):
    """Per-string features, batch predict, cascade filtering, no CUDA."""
    ordered = _calibrate_models(models, seq, sites, desc, cancel)
    if cancel.is_set():
        return 0, 0

    total = 4 ** len(sites)
    positives = 0
    processed = 0
    start = time.perf_counter()

    for chunk in _enumerate_string_chunks(seq, sites, cancel):
        if cancel.is_set():
            break
        B = len(chunk)
        surviving = np.arange(B)
        for _, model, mer, fname, k_list in ordered:
            if cancel.is_set():
                break
            if len(surviving) == 0:
                break
            X = _build_feat_matrix_from_strings(
                [chunk[i] for i in surviving], desc, k_list
            )
            preds, _ = _predict_cpu(model, X)
            mask = preds >= 0.5
            surviving = surviving[mask]

        positives += len(surviving)
        processed += B
        sys.stderr.write(_progress_line(processed, total, start))
        sys.stderr.flush()

    elapsed = time.perf_counter() - start
    return elapsed, positives


def benchmark_cuda(predictor, seq, desc, sites, cancel):
    """Per-string features, batch predict with CUDA, no cascade."""
    active = _get_active_models(predictor.models)
    total = 4 ** len(sites)
    positives = 0
    processed = 0
    start = time.perf_counter()

    for chunk in _enumerate_string_chunks(seq, sites, cancel):
        if cancel.is_set():
            break
        B = len(chunk)
        all_positive = np.ones(B, dtype=bool)
        for _, model, mer, _ in active:
            X = _build_feat_matrix_from_strings(chunk, desc, MER_K_MAP[mer])
            preds, _ = predictor._predict_batch(model, X)
            all_positive &= (preds >= 0.5)
        positives += int(all_positive.sum())
        processed += B
        sys.stderr.write(_progress_line(processed, total, start))
        sys.stderr.flush()

    elapsed = time.perf_counter() - start
    return elapsed, positives


def benchmark_no_batch(models, seq, desc, sites, cancel):
    """Per-string features, per-sample predict (truly naive baseline)."""
    active = _get_active_models(models)
    bases = ["A", "T", "G", "C"]
    seq_list = list(seq)
    total = 4 ** len(sites)
    positives = 0
    processed = 0
    start = time.perf_counter()
    last_print = 0.0

    for combo in product(bases, repeat=len(sites)):
        if cancel.is_set():
            break
        mutant = seq_list.copy()
        for pos, base in zip(sites, combo):
            mutant[pos] = base
        mutant_seq = "".join(mutant)

        all_positive = True
        for _, model, mer, _ in active:
            feat = build_feature_vector_fast(mutant_seq, desc, MER_K_MAP[mer])
            pred, _ = _predict_cpu(model, feat.reshape(1, -1))
            if pred[0] < 0.5:
                all_positive = False
        if all_positive:
            positives += 1

        processed += 1
        now = time.perf_counter()
        if now - last_print >= 0.5 or processed == total:
            sys.stderr.write(_progress_line(processed, total, start))
            sys.stderr.flush()
            last_print = now

    elapsed = time.perf_counter() - start
    return elapsed, positives


def benchmark_full(predictor, seq, smiles, sites, cancel):
    """All optimizations: vectorized + cascade + CUDA."""
    total = 4 ** len(sites)
    start = time.perf_counter()
    last_print = [0.0]

    def on_progress(done, _total, _info):
        now = time.perf_counter()
        if now - last_print[0] >= 0.5 or done == _total:
            sys.stderr.write(_progress_line(done, _total, start))
            sys.stderr.flush()
            last_print[0] = now

    try:
        results = predictor.predict_mutation_batch(
            seq, smiles, sites,
            progress_callback=on_progress,
            should_cancel=lambda: cancel.is_set(),
            collect_results=True,
        )
    except Exception:
        results = None

    elapsed = time.perf_counter() - start
    positives = len(results) if results else 0
    return elapsed, positives


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _compute_sites(sequence: str, n_sites: int) -> list[int]:
    start = (len(sequence) - n_sites) // 2
    return list(range(start, start + n_sites))


def _format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "  TIMEOUT"
    if seconds < 1:
        return f"{seconds * 1000:7.1f}ms"
    if seconds < 60:
        return f"{seconds:7.2f} s"
    m, s = divmod(seconds, 60)
    return f"{m:.0f}m{s:04.1f}s"


def _run_with_timeout(fn, args, timeout, cancel):
    result = [None, 0]
    error = [None]

    def target():
        try:
            elapsed, positives = fn(*args)
            result[0] = elapsed
            result[1] = positives
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        cancel.set()
        t.join(timeout=5)
        return None, 0

    if error[0] is not None:
        raise error[0]

    return result[0], result[1]


def run_benchmarks(
    model_dir: str,
    smiles: str = DEFAULT_SMILES,
    sequence: str = DEFAULT_SEQUENCE,
    site_counts: Optional[list[int]] = None,
    timeout: Optional[float] = 300,
    output_csv: Optional[str] = None,
):
    if site_counts is None:
        site_counts = DEFAULT_SITE_COUNTS

    seq = rna_to_dna(sequence).upper()
    print(f"Sequence : {seq}")
    print(f"SMILES   : {smiles}")
    print(f"Model dir: {model_dir}")
    print()

    print("Loading models...")
    predictor = EnsemblePredictor(model_dir)
    print()

    has_cuda = predictor._device == "cuda"
    if has_cuda:
        print("CUDA detected: using GPU for cuda/full modes")
    else:
        print("No CUDA detected: cuda/full modes will use CPU")

    # Prepare CPU-only models
    cpu_models = []
    try:
        import torch
        for model, mer, fname in predictor.models:
            if isinstance(model, torch.nn.Module):
                model = model.cpu()
            cpu_models.append((model, mer, fname))
    except ImportError:
        cpu_models = list(predictor.models)

    desc = molecular_descriptors(smiles)
    all_results: list[BenchResult] = []
    timed_out: set[str] = set()  # modes that already timed out at smaller site counts
    slow_modes: set[str] = set()  # modes whose elapsed > timeout/4 at smaller site counts

    csv_f, csv_writer = (None, None)
    if output_csv:
        csv_f, csv_writer = _open_csv(output_csv)
        print(f"Results streaming to {output_csv}")

    try:
        for n_sites in site_counts:
            sites = _compute_sites(seq, n_sites)
            n_candidates = 4 ** n_sites
            print(f"--- {n_sites} sites ({n_candidates:,} candidates) ---")
            print(f"    Sites: {sites}")

            def _run_mode(mode, label, fn, args):
                if mode in timed_out:
                    print(f"  {label} ... SKIP (timed out at fewer sites)")
                    r = BenchResult(mode, n_sites, n_candidates)
                    all_results.append(r)
                    if csv_writer: _write_csv_row(csv_writer, csv_f, r)
                    return
                if mode in slow_modes:
                    print(f"  {label} ... SKIP (too slow at fewer sites)")
                    r = BenchResult(mode, n_sites, n_candidates)
                    all_results.append(r)
                    if csv_writer: _write_csv_row(csv_writer, csv_f, r)
                    return
                cancel = threading.Event()
                print(f"  {label} ...")
                try:
                    elapsed, pos = _run_with_timeout(fn, args, timeout, cancel)
                    if elapsed is None:
                        timed_out.add(mode)
                    elif elapsed > timeout / 4:
                        slow_modes.add(mode)
                    print(f"\r  {label} ... {_format_time(elapsed)}  ({pos} positives)           ")
                except Exception as e:
                    elapsed, pos = None, 0
                    timed_out.add(mode)
                    print(f"\r  {label} ... ERROR: {e}")
                r = BenchResult(mode, n_sites, n_candidates, elapsed, pos)
                all_results.append(r)
                if csv_writer: _write_csv_row(csv_writer, csv_f, r)

            _run_mode("original", "Original", benchmark_original,
                       (cpu_models, seq, desc, sites, threading.Event()))
            _run_mode("vectorized", "Vectorized", benchmark_vectorized,
                       (cpu_models, seq, desc, sites, threading.Event()))
            _run_mode("cascade", "Cascade", benchmark_cascade,
                       (cpu_models, seq, desc, sites, threading.Event()))

            if has_cuda:
                _run_mode("cuda", "CUDA", benchmark_cuda,
                           (predictor, seq, desc, sites, threading.Event()))
            else:
                print("  CUDA ... SKIPPED (no CUDA)")
                r = BenchResult("cuda", n_sites, n_candidates)
                all_results.append(r)
                if csv_writer: _write_csv_row(csv_writer, csv_f, r)

            _run_mode("full", "Full", benchmark_full,
                       (predictor, seq, smiles, sites, threading.Event()))

            print()

        _print_table(all_results, site_counts)

        # --- Batch vs No-batch comparison (sites 7-9 only) ---
        batch_sites = [n for n in site_counts if n <= 9]
        if batch_sites:
            print("\n=== Batch vs No-batch (per-string features, CPU) ===")
            batch_results: list[BenchResult] = []
            for n_sites in batch_sites:
                sites = _compute_sites(seq, n_sites)
                n_candidates = 4 ** n_sites
                print(f"--- {n_sites} sites ({n_candidates:,} candidates) ---")

                # No batch
                cancel = threading.Event()
                print(f"  No batch ...")
                try:
                    elapsed, pos = _run_with_timeout(
                        benchmark_no_batch, (cpu_models, seq, desc, sites, cancel),
                        timeout, cancel)
                    print(f"\r  No batch ... {_format_time(elapsed)}  ({pos} positives)           ")
                except Exception as e:
                    elapsed, pos = None, 0
                    print(f"\r  No batch ... ERROR: {e}")
                r = BenchResult("no_batch", n_sites, n_candidates, elapsed, pos)
                batch_results.append(r)
                if csv_writer: _write_csv_row(csv_writer, csv_f, r)

                # Batch (reuse original mode = per-string features + batch predict)
                cancel = threading.Event()
                print(f"  Batch    ...")
                try:
                    elapsed, pos = _run_with_timeout(
                        benchmark_original, (cpu_models, seq, desc, sites, cancel),
                        timeout, cancel)
                    print(f"\r  Batch    ... {_format_time(elapsed)}  ({pos} positives)           ")
                except Exception as e:
                    elapsed, pos = None, 0
                    print(f"\r  Batch    ... ERROR: {e}")
                r = BenchResult("batch", n_sites, n_candidates, elapsed, pos)
                batch_results.append(r)
                if csv_writer: _write_csv_row(csv_writer, csv_f, r)

                if elapsed is not None and batch_results[-2].elapsed is not None:
                    speedup = batch_results[-2].elapsed / elapsed
                    print(f"  Speedup  : {speedup:.1f}x")

            _print_batch_table(batch_results, batch_sites)
    finally:
        if csv_f:
            csv_f.close()


def _print_table(results, site_counts):
    header = (
        f"{'Sites':>5} | {'Candidates':>12} |"
        f" {'Original':>10} | {'VecOnly':>10} |"
        f" {'Cascade':>10} | {'CUDA':>10} | {'Full':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for n in site_counts:
        row_results = {r.mode: r for r in results if r.n_sites == n}
        n_cand = 4 ** n
        parts = [f"{n:>5} | {n_cand:>12,} |"]
        for mode in MODE_NAMES:
            r = row_results.get(mode)
            if r is None or r.elapsed is None:
                parts.append(f" {'TIMEOUT':>10} |")
            else:
                parts.append(f" {_format_time(r.elapsed):>10} |")
        print("".join(parts))
    print(sep)


def _print_batch_table(results, site_counts):
    header = f"{'Sites':>5} | {'Candidates':>12} | {'No Batch':>10} | {'Batch':>10} | {'Speedup':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for n in site_counts:
        row = {r.mode: r for r in results if r.n_sites == n}
        nb = row.get("no_batch")
        b = row.get("batch")
        nb_s = _format_time(nb.elapsed) if nb and nb.elapsed is not None else "  TIMEOUT"
        b_s = _format_time(b.elapsed) if b and b.elapsed is not None else "  TIMEOUT"
        if nb and b and nb.elapsed and b.elapsed:
            spd = f"{nb.elapsed / b.elapsed:.1f}x"
        else:
            spd = "   --"
        print(f"{n:>5} | {4 ** n:>12,} | {nb_s:>10} | {b_s:>10} | {spd:>8}")
    print(sep)


def _open_csv(path):
    """Open CSV file, write header, return (file, writer)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["n_sites", "n_candidates", "mode", "elapsed_seconds", "n_positives"])
    f.flush()
    return f, writer


def _write_csv_row(writer, f, r):
    """Write and flush a single result row."""
    writer.writerow([
        r.n_sites, r.n_candidates, r.mode,
        f"{r.elapsed:.4f}" if r.elapsed is not None else "TIMEOUT",
        r.n_positives,
    ])
    f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    default_model_dir = os.path.join(_PROJECT_ROOT, "models")

    parser = argparse.ArgumentParser(description="Benchmark mutation search optimizations")
    parser.add_argument("--model-dir", default=default_model_dir)
    parser.add_argument("--smiles", default=DEFAULT_SMILES)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--sites", type=int, nargs="+", default=DEFAULT_SITE_COUNTS)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", "-o", default=None)

    args = parser.parse_args()

    run_benchmarks(
        model_dir=args.model_dir,
        smiles=args.smiles,
        sequence=args.sequence,
        site_counts=sorted(args.sites),
        timeout=args.timeout if args.timeout > 0 else None,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
