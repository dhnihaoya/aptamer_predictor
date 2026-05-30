"""Input screen — sequence + target molecule entry."""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Input, Label

from rdkit import Chem

from aptamer_predictor.features import rna_to_dna
from aptamer_predictor.paths import resolve_model_dir


def _resolve_name_or_smiles(text: str) -> tuple[str | None, str | None]:
    """Try to resolve input as SMILES or PubChem name.

    Returns (smiles, display_name) or (None, error_message).
    """
    text = text.strip()
    if not text:
        return None, "Target molecule is required"

    # Try as SMILES first (suppress parse warnings for non-SMILES names)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(text)
    RDLogger.EnableLog("rdApp.error")
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        return smiles, text

    # Try PubChem name resolution
    import json
    import urllib.request
    import urllib.error

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(text)}/property/IsomericSMILES/JSON"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("SMILES")
        return smiles, text
    except (urllib.error.URLError, KeyError, IndexError):
        return None, f"Cannot resolve '{text}' as SMILES or molecule name"


def _default_result_filename() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def _normalize_result_filename(text: str) -> tuple[str | None, str | None]:
    filename = text.strip()
    if not filename:
        return None, "Result filename is required"

    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"

    invalid_chars = set('/\\:*?"<>|')
    if any(char in invalid_chars for char in filename):
        return None, "Result filename contains invalid characters"

    if filename in {".csv", "..csv"}:
        return None, "Result filename is invalid"

    return filename, None


class InputScreen(Screen):
    """Screen for entering aptamer sequence and target molecule."""

    def compose(self) -> ComposeResult:
        with Container(id="input-container"):
            yield Label("Aptamer Mutation Predictor", classes="title")
            yield Label("Sequence (DNA/RNA):")
            yield Input(
                placeholder="e.g. GGGAGATTACGC...",
                id="seq-input",
            )
            yield Label("Target (SMILES or molecule name):")
            yield Input(
                placeholder="e.g. theophylline or CN1C=NC2=C1...",
                id="smiles-input",
            )
            yield Label("Result filename:")
            yield Input(
                value=_default_result_filename(),
                placeholder="e.g. 茶碱结果.csv",
                id="result-filename-input",
            )
            yield Label("", id="error")
            with Container(classes="button-row"):
                yield Button("Resolve & Continue", variant="primary", id="continue-btn")
            yield Label("", id="status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "continue-btn":
            return

        seq_input = self.query_one("#seq-input", Input)
        smi_input = self.query_one("#smiles-input", Input)
        filename_input = self.query_one("#result-filename-input", Input)
        error_label = self.query_one("#error", Label)
        status_label = self.query_one("#status", Label)

        sequence = seq_input.value.strip().upper()
        target = smi_input.value.strip()
        result_filename, filename_error = _normalize_result_filename(
            filename_input.value
        )

        # Validate sequence
        if not sequence:
            error_label.update("Sequence is required")
            return
        valid_bases = set("ATGCU")
        if not all(c in valid_bases for c in sequence):
            error_label.update(
                f"Invalid bases: {set(c for c in sequence if c not in valid_bases)}"
            )
            return
        if filename_error is not None:
            error_label.update(filename_error)
            return

        error_label.update("")
        status_label.update("Resolving molecule...")

        # Resolve molecule (blocking but fast enough for single call)
        import urllib.parse
        result = _resolve_name_or_smiles(target)
        smiles, info = result

        if smiles is None:
            error_label.update(info)
            status_label.update("")
            return

        status_label.update(f"Resolved: {info} -> {smiles[:40]}")

        # Store in app state
        app = self.app
        app.sequence = rna_to_dna(sequence)
        app.smiles = smiles
        app.resolved_name = info
        app.result_filename = result_filename or _default_result_filename()

        # Load predictor lazily
        if app.predictor is None:
            status_label.update("Loading models...")
            model_dir = resolve_model_dir(app.model_dir)

            from aptamer_predictor.predictor import EnsemblePredictor
            try:
                app.predictor = EnsemblePredictor(model_dir)
            except FileNotFoundError as exc:
                error_label.update(str(exc))
                status_label.update("")
                return

        # Navigate to mutation screen
        from ..screens.mutation_screen import MutationScreen
        app.push_screen(MutationScreen(sequence))
