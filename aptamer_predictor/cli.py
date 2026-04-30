"""Command-line interface for aptamer-small molecule interaction prediction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Optional

import numpy as np

from aptamer_predictor.paths import resolve_model_dir


def _resolve_model_dir(args_model_dir: Optional[str]) -> str:
    """Determine the model directory: CLI arg > env var > default."""
    return resolve_model_dir(args_model_dir)


# ---------------------------------------------------------------------------
# predict sub-command
# ---------------------------------------------------------------------------

def cmd_predict(args):
    """Predict aptamer-small molecule interactions."""
    from aptamer_predictor.predictor import EnsemblePredictor

    model_dir = _resolve_model_dir(args.model_dir)
    print(f"Loading models from: {model_dir}")
    predictor = EnsemblePredictor(model_dir)
    print(f"Loaded {len(predictor.models)} models.\n")

    # --- Single pair mode ---
    if args.aptamer and args.smiles:
        from aptamer_predictor.features import MER_K_MAP, build_feature_vector

        seq = args.aptamer.strip()
        smi = args.smiles.strip()

        print(f"Sequence : {seq[:60]}{'...' if len(seq)>60 else ''}")
        print(f"SMILES   : {smi[:60]}{'...' if len(smi)>60 else ''}\n")

        individual = {}
        labels = []

        for model, mer, fname in predictor.models:
            if mer is None or mer not in MER_K_MAP:
                continue
            feat = build_feature_vector(seq, smi, MER_K_MAP[mer])
            pred = model.predict(feat.reshape(1, -1))[0]
            prob = model.predict_proba(feat.reshape(1, -1))[0, 1]
            label_str = "Binding" if pred == 1 else "Non-binding"
            print(f"  {fname:40s}  {label_str:12s}  P={prob:.4f}")
            individual[fname] = {"label": int(
                pred), "probability": round(float(prob), 6)}
            labels.append(int(pred))

        # Ensemble label: Binding only if all 9 models predict Binding, otherwise Non-binding
        ens_label = "Binding" if all(label == 1 for label in labels) else "Non-binding"
        print(f"\n  {'Ensemble':40s}  {ens_label:12s}")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({
                    "sequence": seq,
                    "smiles": smi,
                    "individual": individual,
                    "ensemble_label": 1 if all(label == 1 for label in labels) else 0,
                }, f, indent=2, ensure_ascii=False)
            print(f"\nResult saved to {args.output}")
        return

    # --- Batch mode from CSV ---
    if args.input:
        input_path = args.input
        if not os.path.isfile(input_path):
            print(
                f"Error: input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Detect CSV format by reading header
        with open(input_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

        # Flexible column detection
        header_lower = [h.strip().lower() for h in header]
        seq_col = _find_col(
            header_lower, ["aptamer", "sequence", "seq", "aptamer sequence"])
        smi_col = _find_col(header_lower, ["smiles"])
        label_col = _find_col(
            header_lower, ["label", "true_label", "true label"])
        id_col = _find_col(
            header_lower, ["id", "pubchem_id", "pubchem id", "pubchemid"])

        if seq_col is None or smi_col is None:
            print(
                "Error: CSV must contain at least 'sequence' and 'smiles' columns.\n"
                f"Found headers: {header}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Input file : {input_path}")
        print(f"Columns    : sequence={header[seq_col]}, smiles={header[smi_col]}"
              + (f", label={header[label_col]}" if label_col is not None else "")
              + (f", id={header[id_col]}" if id_col is not None else "")
              + "\n")

        # Read data
        sequences, smiles_list, labels, ids = [], [], [], []
        skipped = 0
        with open(input_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                max_col = max(seq_col, smi_col)
                if label_col is not None:
                    max_col = max(max_col, label_col)
                if id_col is not None:
                    max_col = max(max_col, id_col)
                if len(row) <= max_col:
                    skipped += 1
                    continue
                seq_val = row[seq_col].strip()
                smi_val = row[smi_col].strip()
                if not seq_val or not smi_val:
                    skipped += 1
                    continue
                sequences.append(seq_val)
                smiles_list.append(smi_val)
                labels.append(
                    int(row[label_col]) if label_col is not None and row[label_col] else None)
                ids.append(row[id_col] if id_col is not None else None)

        if skipped:
            print(f"Skipped {skipped} empty row(s).\n")

        print(f"Total samples: {len(sequences)}\n")

        # Predict
        results = predictor.predict_batch(
            sequences, smiles_list,
            labels=labels if any(l is not None for l in labels) else None,
            ids=ids if any(i is not None for i in ids) else None,
        )

        # Output
        output_path = args.output or "predictions.csv"
        _write_batch_results(results, output_path)
        print(f"\nPredictions saved to {output_path}")
        return

    print("Error: provide --aptamer and --smiles for single prediction, "
          "or --input for batch prediction.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# evaluate sub-command
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    """Evaluate models on pre-formatted test data."""
    from aptamer_predictor.predictor import EnsemblePredictor

    model_dir = _resolve_model_dir(args.model_dir)
    data_dir = args.data_dir
    output_dir = args.output_dir or "./predictions"

    print(f"Models     : {model_dir}")
    print(f"Test data  : {data_dir}")
    print(f"Output     : {output_dir}\n")

    predictor = EnsemblePredictor(model_dir)
    print(f"Loaded {len(predictor.models)} models.\n")

    predictor.evaluate(data_dir, output_dir)


# ---------------------------------------------------------------------------
# extract sub-commands
# ---------------------------------------------------------------------------

def cmd_extract_aptamer(args):
    """Extract k-mer frequency features from aptamer sequences."""
    from aptamer_predictor.features import kmer_features, descriptor_names, rna_to_dna

    input_path = args.input
    output_path = args.output
    k_list = sorted(args.k)

    print(f"Input : {input_path}")
    print(f"k-mer : {k_list}")

    # Read input CSV (expects: RNA sequence, DNA sequence — or just sequence)
    rows_out = []
    header_fields = None

    with open(input_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Determine which column(s) to use
        # Try to find sequence columns
        seq_col = _find_col(
            [h.strip().lower() for h in header],
            ["aptamer seqtence(u-t)", "dna sequence",
             "aptamer sequence", "sequence", "seq"]
        )
        if seq_col is None:
            seq_col = 1 if len(header) > 1 else 0

        # Build header
        from aptamer_predictor.features import _get_all_kmers
        kmer_names = []
        for k in k_list:
            kmer_names.extend(_get_all_kmers(k))
        header_fields = ["sequence"] + kmer_names

        for row in reader:
            seq = row[seq_col].strip()
            seq = rna_to_dna(seq)
            feats = kmer_features(row[seq_col], k_list)
            rows_out.append([seq] + feats)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_fields)
        writer.writerows(rows_out)

    print(
        f"Output: {output_path}  ({len(rows_out)} sequences, {len(kmer_names)} features)")


def cmd_extract_molecule(args):
    """Extract RDKit molecular descriptors from SMILES."""
    from aptamer_predictor.features import molecular_descriptors, descriptor_names

    input_path = args.input
    output_path = args.output

    desc_names = descriptor_names()

    rows_out = []
    with open(input_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find SMILES column
        smi_col = _find_col(
            [h.strip().lower() for h in header],
            ["smiles"]
        )
        if smi_col is None:
            smi_col = 3  # default for Non-redundant target.csv

        # Also get pubchem_id, target, cas if available
        id_col = _find_col([h.strip().lower() for h in header], ["pubchem id"])
        target_col = _find_col([h.strip().lower() for h in header], ["target"])
        cas_col = _find_col([h.strip().lower() for h in header], ["cas"])

        for row in reader:
            smiles = row[smi_col]
            descs = molecular_descriptors(smiles)
            prefix = []
            if id_col is not None:
                prefix.append(row[id_col])
            if target_col is not None:
                prefix.append(row[target_col])
            if cas_col is not None:
                prefix.append(row[cas_col])
            prefix.append(smiles)
            rows_out.append(prefix + descs)

    out_header = []
    if id_col is not None:
        out_header.append("pubchem_id")
    if target_col is not None:
        out_header.append("target")
    if cas_col is not None:
        out_header.append("cas")
    out_header.append("smiles")
    out_header.extend(desc_names)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(rows_out)

    print(
        f"Output: {output_path}  ({len(rows_out)} molecules, {len(desc_names)} descriptors)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(header_lower: list[str], candidates: list[str]) -> int | None:
    """Find the first matching column index (case-insensitive)."""
    for candidate in candidates:
        for i, h in enumerate(header_lower):
            if h == candidate:
                return i
    return None


def _write_batch_results(results: list[dict], output_path: str):
    """Write batch prediction results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "smiles", "ensemble_label"])
        for r in results:
            writer.writerow([
                r.get("sequence", ""),
                r.get("smiles", ""),
                r["ensemble_label"],
            ])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptamer-predictor",
        description="Aptamer-Small Molecule Interaction Prediction Tool",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Path to directory containing .pkl model files "
             "(default: models/)",
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="Launch interactive Textual TUI",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- predict ---
    p_pred = sub.add_parser(
        "predict", help="Predict aptamer-small molecule interactions")
    p_pred.add_argument(
        "--aptamer", help="Single aptamer sequence (RNA or DNA)")
    p_pred.add_argument("--smiles", help="Single SMILES string")
    p_pred.add_argument(
        "--input", "-i", help="Input CSV file for batch prediction")
    p_pred.add_argument("--output", "-o", help="Output file path")

    # --- evaluate ---
    p_eval = sub.add_parser(
        "evaluate", help="Evaluate models on pre-formatted test data")
    p_eval.add_argument("--data-dir", required=True,
                        help="Directory with test CSV files")
    p_eval.add_argument(
        "--output-dir", default="./predictions", help="Output directory")

    # --- extract-aptamer ---
    p_ext_a = sub.add_parser(
        "extract-aptamer", help="Extract k-mer features from aptamer sequences")
    p_ext_a.add_argument("--input", "-i", required=True, help="Input CSV file")
    p_ext_a.add_argument("--output", "-o", required=True,
                         help="Output CSV file")
    p_ext_a.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4],
                         help="k values for k-mer extraction (default: 1 2 3 4)")

    # --- extract-molecule ---
    p_ext_m = sub.add_parser(
        "extract-molecule", help="Extract molecular descriptors from SMILES")
    p_ext_m.add_argument("--input", "-i", required=True, help="Input CSV file")
    p_ext_m.add_argument("--output", "-o", required=True,
                         help="Output CSV file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.tui:
        from aptamer_predictor.tui.app import run_tui
        run_tui(model_dir=_resolve_model_dir(args.model_dir))
        return

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
        "extract-aptamer": cmd_extract_aptamer,
        "extract-molecule": cmd_extract_molecule,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
