"""Results screen — progress bar + streaming results table + auto CSV export."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime

import textual.worker as textual_worker
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, ProgressBar
from textual.worker import Worker

from aptamer_predictor.predictor import PredictionCancelled


class ResultsScreen(Screen):
    """Screen showing prediction progress and results."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prediction_worker: Worker | None = None
        self._cancel_event = threading.Event()
        self._prediction_done = False
        self._return_to_input_pending = False
        self._is_unmounted = False
        self._csv_file = None
        self._csv_writer = None
        self._csv_lock = threading.Lock()
        self._csv_filename: str = ""
        self._hit_count = 0
        self._start_time = 0.0
        self._total_mutations = 0
        self._last_progress_done = 0
        self._last_progress_time = 0.0
        self._estimated_done = 0.0
        self._progress_timer = None
        self._recent_results: list[dict] = []
        self._row_key_counter = 0

    def compose(self) -> ComposeResult:
        with Container(id="results-container"):
            yield Label("Prediction Results", classes="title")
            yield ProgressBar(total=100, id="progress-bar")
            yield Label("Preparing...", id="progress-label")
            yield Label("Hits: 0", id="hit-counter")
            yield DataTable(id="results-table")
            with Container(classes="button-row"):
                yield Button("New Search", variant="default", id="new-btn")

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Sequence", "Prob", "Time")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_filename = getattr(self.app, "result_filename", "").strip()
        self._csv_filename = requested_filename or f"{ts}.csv"
        self._csv_file = open(self._csv_filename, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        # Build CSV header: sequence + per-model probabilities + mean_probability
        app = self.app
        assert app.predictor is not None
        model_names = [
            fname for (_, _, fname) in getattr(app.predictor, "models", [])
        ]
        self._model_names = model_names
        self._csv_writer.writerow(
            ["sequence"]
            + [f"prob_{name}" for name in model_names]
            + ["mean_probability"]
        )
        self._csv_file.flush()

        app = self.app
        assert app.predictor is not None
        self._total_mutations = 4 ** len(app.selected_sites)
        self._start_time = time.monotonic()
        self.query_one("#progress-bar", ProgressBar).total = self._total_mutations
        self._progress_timer = self.set_interval(1.0, self._tick_progress)
        self._update_progress(0, self._total_mutations, 0.0, None)

        self._prediction_worker = self.run_worker(
            self._run_prediction,
            name="prediction",
            thread=True,
        )

    def _run_prediction(self) -> None:
        app = self.app
        predictor = app.predictor
        assert predictor is not None
        get_current_worker = getattr(textual_worker, "get_current_worker", None)
        worker = get_current_worker() if callable(get_current_worker) else None

        sequence = app.sequence
        smiles = app.smiles
        sites = app.selected_sites

        total_mutations = 4 ** len(sites)
        batch_size = max(50, total_mutations // 200)

        def on_result(result: dict) -> None:
            with self._csv_lock:
                if self._csv_writer is None or self._csv_file is None:
                    return
                model_probs = result.get("model_probabilities", [])
                row = [result["sequence"]]
                row.extend(f"{p:.6f}" for p in model_probs)
                row.append(f"{result['mean_probability']:.6f}")
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            self._hit_count += 1
            seen_at = datetime.now().strftime("%H:%M")
            self.app.call_from_thread(self._update_hit_counter)
            self.app.call_from_thread(self._append_recent_result, result, seen_at)

        def on_progress(done: int, total: int, info: dict) -> None:
            del info
            now = time.monotonic()
            self._last_progress_done = done
            self._last_progress_time = now
            self._estimated_done = float(done)
            pct = (done / total * 100) if total else 0.0
            self.app.call_from_thread(self._update_progress, done, total, pct, self._eta_seconds())

        try:
            predictor.predict_mutation_batch(
                sequence,
                smiles,
                sites,
                batch_size=batch_size,
                progress_callback=on_progress,
                should_cancel=lambda: self._should_cancel(worker),
                result_callback=on_result,
                collect_results=False,
            )
        except PredictionCancelled:
            self.app.call_from_thread(self._handle_prediction_cancelled)
            return
        except Exception as exc:
            self.app.call_from_thread(self._show_error, str(exc))
            return
        finally:
            self._close_csv()

        self.app.call_from_thread(self._show_results)

    def _should_cancel(self, worker: Worker | None) -> bool:
        if self._cancel_event.is_set():
            return True

        if worker is None:
            return False

        is_cancelled = getattr(worker, "is_cancelled", None)
        if callable(is_cancelled):
            return bool(is_cancelled())
        if is_cancelled is not None:
            return bool(is_cancelled)
        return False

    def _cancel_prediction(self) -> None:
        self._cancel_event.set()
        self.query_one("#progress-label", Label).update("Cancelling...")
        self.query_one("#new-btn", Button).disabled = True

        worker = self._prediction_worker
        if worker is not None:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()

    def _return_to_input(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def _set_label(self, text: str) -> None:
        if self._cancel_event.is_set() or self._is_unmounted:
            return
        self.query_one("#progress-label", Label).update(text)

    def _format_speed(self, speed: float) -> str:
        if speed >= 1_000_000:
            return f"{speed / 1_000_000:.1f}M/s"
        if speed >= 1_000:
            return f"{speed / 1_000:.1f}K/s"
        return f"{speed:.0f}/s"

    def _update_progress(
        self,
        done: float,
        total: int,
        pct: float,
        eta_seconds: float | None,
    ) -> None:
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=done)
        elapsed = time.monotonic() - self._start_time if self._start_time else 1.0
        speed = done / elapsed if elapsed > 0 else 0.0
        eta_text = self._format_eta(eta_seconds)
        self._set_label(
            f"{pct:.1f}% | {int(done):,} / {total:,} | {self._format_speed(speed)} | ETA: {eta_text} → {self._csv_filename}"
        )

    def _tick_progress(self) -> None:
        if self._prediction_done or self._cancel_event.is_set():
            return

        estimated_done = float(self._last_progress_done)
        if self._last_progress_done > 0 and self._last_progress_time > self._start_time:
            elapsed_since_callback = max(0.0, time.monotonic() - self._last_progress_time)
            average_speed = self._last_progress_done / max(
                1e-9,
                self._last_progress_time - self._start_time,
            )
            estimated_done += average_speed * elapsed_since_callback

        self._estimated_done = min(float(self._total_mutations), max(float(self._last_progress_done), estimated_done))
        pct = (self._estimated_done / self._total_mutations * 100) if self._total_mutations else 0.0
        self._update_progress(self._estimated_done, self._total_mutations, pct, self._eta_seconds())

    def _eta_seconds(self) -> float | None:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        speed = self._estimated_done / elapsed if elapsed > 0 else 0.0
        if speed <= 0 or self._estimated_done <= 0:
            return None
        return max(0.0, (self._total_mutations - self._estimated_done) / speed)

    def _format_eta(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        remaining = max(0, int(seconds))
        minutes, secs = divmod(remaining, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _update_hit_counter(self) -> None:
        if self._cancel_event.is_set() or self._is_unmounted:
            return
        self.query_one("#hit-counter", Label).update(
            f"Hits: {self._hit_count:,}"
        )

    def _append_recent_result(self, result: dict, seen_at: str) -> None:
        table = self.query_one("#results-table", DataTable)
        entry = {
            "sequence": result["sequence"],
            "mean_probability": result["mean_probability"],
            "time": seen_at,
            "row_key": f"recent-{self._row_key_counter}",
        }
        self._row_key_counter += 1
        self._recent_results.append(entry)
        table.add_row(
            entry["sequence"],
            f"{entry['mean_probability']:.4f}",
            entry["time"],
            key=entry["row_key"],
        )

        if len(self._recent_results) > 10:
            evicted = self._recent_results.pop(0)
            table.remove_row(evicted["row_key"])

    def _show_results(self) -> None:
        if self._cancel_event.is_set():
            self._handle_prediction_cancelled()
            return

        self._prediction_done = True
        if self._progress_timer is not None:
            self._progress_timer.stop()
        self.query_one("#progress-label", Label).update(
            f"Done — {self._hit_count:,} candidates → {self._csv_filename}"
        )
        self.query_one("#new-btn", Button).disabled = False

    def _show_error(self, message: str) -> None:
        self._prediction_done = True
        self.query_one("#progress-label", Label).update(f"Error: {message}")
        self.query_one("#new-btn", Button).disabled = False

    def _handle_prediction_cancelled(self) -> None:
        self._prediction_done = True
        if self._is_unmounted:
            return

        self.query_one("#progress-label", Label).update("Prediction cancelled")

        if self._return_to_input_pending:
            self._return_to_input()
            return

        self.query_one("#new-btn", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-btn":
            if self._prediction_done:
                self._return_to_input()
                return

            self._return_to_input_pending = True
            self._cancel_prediction()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            with self._csv_lock:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def on_unmount(self) -> None:
        self._is_unmounted = True
        if self._progress_timer is not None:
            self._progress_timer.stop()
        if not self._prediction_done:
            self._cancel_event.set()
            worker = self._prediction_worker
            if worker is not None:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
