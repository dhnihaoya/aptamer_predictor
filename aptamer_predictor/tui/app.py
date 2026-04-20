"""PredictorApp — Textual TUI for aptamer mutation prediction."""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from aptamer_predictor.predictor import EnsemblePredictor

from .screens.input_screen import InputScreen


class PredictorApp(App):
    """Interactive TUI for aptamer-small molecule mutation prediction."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self, model_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.predictor: EnsemblePredictor | None = None
        self.sequence: str = ""
        self.smiles: str = ""
        self.resolved_name: str = ""
        self.result_filename: str = ""
        self.selected_sites: list[int] = []

    def on_mount(self) -> None:
        self.push_screen(InputScreen())

    def set_predictor(self, predictor: EnsemblePredictor) -> None:
        self.predictor = predictor


def run_tui(model_dir: str | None = None) -> None:
    """Public entry point for launching the TUI."""
    app = PredictorApp(model_dir=model_dir)
    app.run()
