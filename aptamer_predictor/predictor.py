"""Model loading, prediction, and ensemble voting."""

from __future__ import annotations

import glob
import os
import pickle
import re
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# PyTorch RNN wrapper  (loaded lazily to avoid hard import)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_nn = None


def _ensure_torch():
    global _TORCH_AVAILABLE, _nn
    if _nn is None:
        try:
            import torch
            import torch.nn as nn
            _nn = nn
            _TORCH_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "PyTorch is required for loading RNN/biRNN models. "
                "Install it with: pip install torch"
            )


class SimpleRNN:
    """PyTorch RNN model wrapper compatible with sklearn-style API.

    Must be defined in __main__ namespace before pickle.load() so that
    the deserializer can find the class.
    """

    # Will be set to the actual nn.Module subclass after _ensure_torch()
    _nn_module = None

    @classmethod
    def _as_module(cls):
        """Return a proper nn.Module subclass that pickle can instantiate."""
        if cls._nn_module is not None:
            return cls._nn_module

        _ensure_torch()
        import torch
        import torch.nn as nn

        class _SimpleRNN(nn.Module):
            """RNN -> Linear -> Sigmoid -> Linear -> Sigmoid."""

            def _to_device(self, x):
                """Move input to same device as model parameters."""
                device = next(self.parameters()).device
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(
                        np.asarray(x, dtype=np.float32), device=device
                    )
                elif x.device != device:
                    x = x.to(device)
                return x

            def forward(self, x):
                x = self._to_device(x)
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # (1, features) — single sample
                # batch_first=True: expect (batch, seq_len, input_size)
                x = x.unsqueeze(1)  # (batch, 1, features)
                out, _ = self.rnn(x)
                out = out.squeeze(1)  # (batch, hidden * num_directions)
                out = self.sig1(self.fc1(out))
                out = self.sig2(self.fc2(out))  # (batch, 1)
                return out.squeeze(-1)  # (batch,)

            def predict_proba(self, X):
                self.eval()
                with torch.no_grad():
                    if not isinstance(X, np.ndarray):
                        X = np.asarray(X, dtype=np.float32)
                    t = torch.FloatTensor(X)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    out = self.forward(t).cpu().numpy()
                    return np.column_stack([1 - out, out])

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] >= 0.5).astype(int)

        cls._nn_module = _SimpleRNN
        return _SimpleRNN


class PredictionCancelled(Exception):
    """Raised when a long-running mutation search is cancelled."""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _extract_mer_label(filename: str) -> Optional[str]:
    """Extract mer label like '1mer', '23mer' from a model filename."""
    m = re.search(r"\((\d+mer)\)", filename)
    return m.group(1) if m else None


def load_model(filepath: str):
    """Load a single .pkl model, handling both sklearn and PyTorch types."""
    mer = _extract_mer_label(os.path.basename(filepath))
    basename = os.path.basename(filepath)

    # Check if this is a PyTorch model by inspecting filename
    is_pytorch = basename.endswith("RNN.pkl") or basename.endswith("biRNN.pkl")

    if is_pytorch:
        _ensure_torch()
        import __main__
        __main__.SimpleRNN = SimpleRNN._as_module()
        with open(filepath, "rb") as f:
            model = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            model = pickle.load(f)

    return model, mer


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """Manage all 9 pre-trained models and perform ensemble prediction."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = []  # list of (model, mer_label, filepath)
        self._device = "cpu"
        self._load_all()
        self._setup_cuda()

    @staticmethod
    def _normalize_pair(sequence: str, smiles: str) -> tuple[str, str]:
        """Normalize external prediction inputs consistently."""
        return sequence.strip(), smiles.strip()

    def _load_all(self):
        pattern = os.path.join(self.model_dir, "(*mer)*.pkl")
        files = sorted(glob.glob(pattern))
        if not files:
            # Also try without parens pattern for directories with spaces
            pattern = os.path.join(self.model_dir, "*.pkl")
            files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No .pkl model files found in {self.model_dir}"
            )

        for fp in files:
            try:
                model, mer = load_model(fp)
                self.models.append((model, mer, os.path.basename(fp)))
                print(f"  Loaded: {os.path.basename(fp)}")
            except Exception as e:
                print(f"  Warning: failed to load {os.path.basename(fp)}: {e}")

        if len(self.models) < 9:
            print(
                f"  Warning: expected 9 models, only loaded "
                f"{len(self.models)}. Ensemble results may be unreliable."
            )

    # ---- CUDA setup ------------------------------------------------------

    def _setup_cuda(self) -> None:
        """Move PyTorch RNN models to CUDA if available."""
        from aptamer_predictor.cuda import get_device
        self._device = get_device()
        if self._device != "cuda":
            return

        import torch
        new_models = []
        for model, mer, fname in self.models:
            if isinstance(model, torch.nn.Module):
                model = model.to("cuda")
            new_models.append((model, mer, fname))
        self.models = new_models
        print(f"  CUDA acceleration enabled ({torch.cuda.get_device_name(0)})")

    @staticmethod
    def _is_xgboost(model) -> bool:
        try:
            from xgboost import XGBClassifier, Booster
            return isinstance(model, (XGBClassifier, Booster))
        except ImportError:
            return False

    def _predict_batch(self, model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction with optional CUDA acceleration.

        Returns (predictions, positive_class_probabilities).
        """
        # XGBoost GPU path
        if self._device == "cuda" and self._is_xgboost(model):
            try:
                import xgboost as xgb
                booster = model.get_booster()
                dm = xgb.DMatrix(X.astype(np.float32), device="cuda")
                probs = booster.predict(dm)
                return (probs >= 0.5).astype(int), probs
            except Exception:
                pass  # fall through to CPU

        # Standard sklearn / PyTorch API (PyTorch handles CUDA internally)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return preds.astype(int), probs

    # ---- single sample ---------------------------------------------------

    def predict_one(self, sequence: str, smiles: str) -> dict:
        """Predict with all models on a single aptamer-SMILES pair.

        Args:
            sequence: Aptamer sequence (RNA or DNA).
            smiles: Small molecule SMILES string.

        Returns:
            dict with keys: individual (dict of model->pred), ensemble_label
        """
        from aptamer_predictor.features import MER_K_MAP, build_feature_vector

        sequence, smiles = self._normalize_pair(sequence, smiles)
        results = {}
        model_labels = []

        for model, mer, fname in self.models:
            if mer is None or mer not in MER_K_MAP:
                continue
            feat = build_feature_vector(sequence, smiles, MER_K_MAP[mer])
            pred = model.predict(feat.reshape(1, -1))[0]
            prob = model.predict_proba(feat.reshape(1, -1))[0, 1]
            results[fname] = {"label": int(pred), "probability": float(prob)}
            model_labels.append(int(pred))

        # Ensemble label: 1 only if all 9 models predict 1, otherwise 0
        ensemble_label = 1 if all(label == 1 for label in model_labels) else 0

        return {
            "individual": results,
            "ensemble_label": ensemble_label,
        }

    # ---- batch prediction with raw aptamer + SMILES ----------------------

    def predict_batch(
        self,
        sequences: list[str],
        smiles_list: list[str],
        labels: Optional[list[int]] = None,
        ids: Optional[list] = None,
    ) -> list[dict]:
        """Full pipeline: extract features → predict with all models → ensemble vote.

        Args:
            sequences: List of aptamer sequences (RNA or DNA).
            smiles_list: List of SMILES strings.
            labels: Optional true labels for evaluation.
            ids: Optional sample IDs.

        Returns:
            List of result dicts.
        """
        from aptamer_predictor.features import MER_K_MAP, build_feature_vector

        all_results = []

        for i, (seq, smi) in enumerate(zip(sequences, smiles_list)):
            seq, smi = self._normalize_pair(seq, smi)
            sample = {
                "sequence": seq,
                "smiles": smi,
            }
            if ids is not None:
                sample["id"] = ids[i]
            if labels is not None:
                sample["true_label"] = labels[i]

            individual = {}
            model_labels = []

            for model, mer, fname in self.models:
                if mer is None or mer not in MER_K_MAP:
                    continue
                k_list = MER_K_MAP[mer]
                feat = build_feature_vector(seq, smi, k_list)

                pred = model.predict(feat.reshape(1, -1))[0]
                prob = model.predict_proba(feat.reshape(1, -1))[0, 1]

                individual[fname] = {
                    "label": int(pred),
                    "probability": round(float(prob), 6),
                }
                model_labels.append(int(pred))

            # Ensemble label: 1 only if all 9 models predict 1, otherwise 0
            ensemble_label = 1 if all(label == 1 for label in model_labels) else 0

            sample["individual"] = individual
            sample["ensemble_label"] = ensemble_label
            all_results.append(sample)

            if (i + 1) % 50 == 0 or i == len(sequences) - 1:
                print(f"  Processed {i + 1}/{len(sequences)} samples")

        return all_results

    # ---- mutation batch prediction (optimized) ---------------------------

    def predict_mutation_batch(
        self,
        base_sequence: str,
        smiles: str,
        sites: list[int],
        *,
        batch_size: int = 2000,
        sub_batch_size: Optional[int] = None,
        progress_callback=None,
        should_cancel: Optional[Callable[[], bool]] = None,
        result_callback: Optional[Callable[[dict], None]] = None,
        collect_results: bool = True,
    ) -> Optional[list[dict]]:
        """Enumerate all mutants at selected sites, batch-predict.

        Collects **only** candidates where all 9 models predict binding
        (ensemble_label == 1).

        Optimizations:
        - Pre-computes molecular descriptors once.
        - Vectorized NumPy batch k-mer + feature matrix.
        - Dynamic calibration determines optimal model order.
        - Early-exit sequential filtering: each model only processes
          survivors from the previous model.

        Args:
            base_sequence: Original aptamer sequence.
            smiles: Target molecule SMILES.
            sites: List of 0-indexed positions to mutate.
            batch_size: Number of candidates per inference chunk.
            sub_batch_size: Internal vectorized processing chunk size.
            progress_callback: Optional callable(done, total, info_dict).
            should_cancel: Optional callable returning True when the
                enumeration should abort immediately.
            result_callback: Optional callable invoked for each positive hit.
            collect_results: When False, stream positives through
                result_callback without accumulating them in memory.

        Returns:
            List of positive-result dicts sorted by mean probability
            (descending), or None when collect_results is False.
        """
        from itertools import product

        from aptamer_predictor.features import (
            _ENCODE_TABLE,
            MER_K_MAP,
            build_feature_matrix,
            molecular_descriptors,
            rna_to_dna,
        )

        base_sequence, smiles = self._normalize_pair(base_sequence, smiles)
        seq = rna_to_dna(base_sequence).upper()
        seq_list = list(seq)
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        bases = ["A", "T", "G", "C"]
        base_bytes = np.frombuffer(b"ATGC", dtype=np.uint8)
        sites_arr = np.array(sites, dtype=np.intp)
        if np.any(sites_arr < 0) or np.any(sites_arr >= len(seq)):
            raise ValueError("Mutation site index out of range")

        batch_size = max(1, int(batch_size))
        if sub_batch_size is None:
            sub_batch_size = min(65536, batch_size)
        sub_batch_size = max(1, int(sub_batch_size))
        total = 4 ** len(sites)

        def _check_cancelled() -> None:
            if should_cancel and should_cancel():
                raise PredictionCancelled()

        # Pre-compute molecular descriptors once
        _check_cancelled()
        desc = molecular_descriptors(smiles)

        # Collect per-model configs
        model_configs = []
        for model, mer, fname in self.models:
            if mer is None or mer not in MER_K_MAP:
                continue
            model_configs.append(
                (len(model_configs), model, mer, fname, MER_K_MAP[mer])
            )

        # --- Calibrate: determine optimal model order via small sample ---
        calib_seqs = []
        for combo in product(bases, repeat=len(sites)):
            _check_cancelled()
            if len(calib_seqs) >= 64:
                break
            mutant = seq_list.copy()
            for pos, new_base in zip(sites, combo):
                if 0 <= pos < len(mutant):
                    mutant[pos] = new_base
            calib_seqs.append("".join(mutant))

        scored: list[tuple[int, int, object, str, str, list[int]]] = []
        for original_idx, model, mer, fname, k_list in model_configs:
            _check_cancelled()
            X = build_feature_matrix(calib_seqs, desc, k_list)
            preds, _ = self._predict_batch(model, X)
            n_pos = int(preds.sum())
            scored.append((n_pos, original_idx, model, mer, fname, k_list))
        scored.sort(key=lambda x: x[0])  # most selective first

        ordered_models = [
            (original_idx, m, mer, fn, kl)
            for _, original_idx, m, mer, fn, kl in scored
        ]

        # --- Main enumeration with early-exit filtering ---
        positives: Optional[list[dict]] = [] if collect_results else None
        done = 0
        progress_mark = 0

        def _flush_chunk(mutant_bytes: np.ndarray) -> None:
            nonlocal done, progress_mark
            _check_cancelled()
            if mutant_bytes.size == 0:
                return

            encoded_mutants = _ENCODE_TABLE[mutant_bytes]
            B = encoded_mutants.shape[0]

            # Early-exit sequential filtering
            surviving = np.arange(B)
            # Store per-candidate probs: list of arrays, one per model
            all_model_probs = np.zeros((B, len(model_configs)), dtype=np.float64)

            for model_idx, model, mer, fname, k_list in ordered_models:
                _check_cancelled()
                if len(surviving) == 0:
                    break

                X = build_feature_matrix(encoded_mutants[surviving], desc, k_list)
                preds, probs = self._predict_batch(model, X)

                # Store probs for all surviving candidates
                all_model_probs[surviving, model_idx] = probs

                # Filter: keep only positives
                mask = preds >= 0.5
                surviving = surviving[mask]

            done += B
            progress_mark += B
            if progress_callback and (progress_mark >= batch_size or done == total):
                _check_cancelled()
                progress_callback(done, total, {})
                progress_mark = 0

            # Final survivors are the positives
            for idx in surviving:
                probs = all_model_probs[idx]
                mean_prob = float(np.mean(probs))
                result = {
                    "sequence": mutant_bytes[idx].tobytes().decode("ascii"),
                    "mean_probability": round(mean_prob, 6),
                    "ensemble_label": 1,
                    "model_probabilities": [round(float(p), 6) for p in probs],
                }
                if result_callback:
                    result_callback(result)
                if positives is not None:
                    positives.append(result)

        if len(sites) == 0:
            _flush_chunk(seq_bytes.reshape(1, -1))
        else:
            for start in range(0, total, sub_batch_size):
                _check_cancelled()
                stop = min(start + sub_batch_size, total)
                batch_len = stop - start
                digits = np.empty((batch_len, len(sites)), dtype=np.int8)
                values = np.arange(start, stop, dtype=np.int64)

                for pos in range(len(sites) - 1, -1, -1):
                    digits[:, pos] = values % 4
                    values //= 4

                mutant_bytes = np.tile(seq_bytes, (batch_len, 1))
                mutant_bytes[:, sites_arr] = base_bytes[digits]
                _flush_chunk(mutant_bytes)

        if positives is None:
            return None

        positives.sort(key=lambda r: r["mean_probability"], reverse=True)
        return positives

    # ---- evaluate on pre-formatted CSV -----------------------------------

    def evaluate(self, data_dir: str, output_dir: str):
        """Evaluate all models on pre-formatted test CSV files.

        Reads test data files matching pattern:
          {data_dir}/{mer}independent_data all des(104-2080).csv

        Writes per-model predictions and summary metrics to output_dir.
        """
        import csv
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        os.makedirs(output_dir, exist_ok=True)

        def calc_specificity(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        metrics_path = os.path.join(output_dir, "all_models_metrics.csv")
        with open(metrics_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow([
                "Model", "AUC", "Accuracy", "F1-score",
                "Precision", "Recall", "Specificity",
            ])

        for model, mer, fname in self.models:
            if mer is None:
                continue

            # Find matching test data file
            pattern = os.path.join(
                data_dir, f"{mer}*independent_data*all*des*.csv"
            )
            data_files = glob.glob(pattern)

            if not data_files:
                print(f"  Skipping {fname}: no test data found for {mer}")
                continue

            data_file = data_files[0]
            print(f"  Evaluating {fname} with {os.path.basename(data_file)}")

            # Read test data
            test_x, label, seq, pubchem_id = [], [], [], []
            with open(data_file, "r") as g:
                g.readline()  # skip header
                for line in g:
                    items = line.strip().split(",")
                    for j in range(3, len(items)):
                        items[j] = float(items[j])
                    test_x.append(items[3:])
                    label.append(items[0])
                    seq.append(items[1])
                    pubchem_id.append(items[2])

            test_x = np.array(test_x)
            label = np.array(label).astype(int)

            # Predict
            pred_y = model.predict(test_x)
            pred_y_prob = model.predict_proba(test_x)[:, 1]

            # Write per-model predictions
            out_name = fname.replace(".pkl", "") + "_predictions.csv"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w") as f:
                f.write("true_label,seq,pubchemid,pred_label,prob\n")
                for u in range(len(seq)):
                    f.write(
                        f"{label[u]},{seq[u]},{pubchem_id[u]},"
                        f"{pred_y[u]},{pred_y_prob[u]}\n"
                    )

            # Compute metrics
            auc = roc_auc_score(label, pred_y_prob)
            accuracy = accuracy_score(label, pred_y)
            precision = precision_score(label, pred_y)
            f1 = f1_score(label, pred_y)
            recall = recall_score(label, pred_y)
            specificity = calc_specificity(label, pred_y)

            with open(metrics_path, "a", newline="") as mf:
                writer = csv.writer(mf)
                writer.writerow([
                    fname.replace(".pkl", ""),
                    f"{auc:.4f}", f"{accuracy:.4f}", f"{f1:.4f}",
                    f"{precision:.4f}", f"{recall:.4f}", f"{specificity:.4f}",
                ])

            print(
                f"    AUC={auc:.4f}  Acc={accuracy:.4f}  "
                f"F1={f1:.4f}  Prec={precision:.4f}  "
                f"Recall={recall:.4f}  Spec={specificity:.4f}"
            )

        print(f"\nMetrics saved to {metrics_path}")
