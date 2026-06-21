# rl/game_logger.py

import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class GameLogger:
    """
    Asynchronous, non-blocking telemetry and game serialization logger.

    Uses an internal worker thread to handle disk writes (JSON) so that
    file I/O overhead never stalls self-play match generation or engine searches.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/self_play_logs",
        log_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(log_dir if log_dir is not None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe queue for telemetry data payloads
        self.log_queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue()

        # Background worker setup
        self.worker_thread = threading.Thread(target=self._io_worker, daemon=True)
        self.is_running = False
        self.start()

    def start(self) -> None:
        """Starts the background I/O writer thread."""
        if not self.is_running:
            self.is_running = True
            self.worker_thread.start()

    def log_game(
        self,
        game_id: str,
        history: List[Dict[str, Any]],
        result: float,
        termination_reason: str
    ) -> None:
        """
        Enqueues a completed game history payload to be written asynchronously.
        """
        payload = {
            "game_id": game_id,
            "result": result,
            "termination": termination_reason,
            "moves": history
        }
        self.log_queue.put(payload)

    def _io_worker(self) -> None:
        """Continuous loop running in a background thread to pull and write data."""
        while self.is_running or not self.log_queue.empty():
            try:
                # Bounded timeout prevents thread hanging during shutdown sequences
                payload = self.log_queue.get(timeout=1.0)

                if payload is None:
                    # Architectural Fix: Must acknowledge the poison pill task
                    # before breaking to avoid locking up queue synchronization trackers.
                    self.log_queue.task_done()
                    break

                self._write_to_disk(payload)
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def _convert_serializable(self, obj: Any) -> Any:
        """
        Architectural Fix: Recursive cleaner that strips out un-serializable
        PyTorch tensors or NumPy arrays from your feature dictionaries.
        """
        if isinstance(obj, dict):
            return {k: self._convert_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_serializable(v) for v in obj]
        return obj

    def _write_to_disk(self, payload: Dict[str, Any]) -> None:
        """Executes the raw physical file dump."""
        game_id = payload["game_id"]
        file_path = self.output_dir / f"{game_id}.json"

        try:
            # Clean tensor weights and feature arrays before converting to raw JSON string strings
            clean_payload = self._convert_serializable(payload)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(clean_payload, f, indent=2)
        except (IOError, TypeError) as e:
            # Fixed: Catching TypeError explicitly protects the worker loop from dying
            # if an unhandled data type bypasses the structural check.
            print(f"[GameLogger Error] Failed writing game {game_id} to disk: {e}")

    def shutdown(self) -> None:
        """Gracefully flushes remaining items in the queue and safely kills the thread."""
        if not self.is_running:
            return

        self.is_running = False
        # Send a "poison pill" to unblock the get() request and stop the loop
        self.log_queue.put(None)
        self.worker_thread.join(timeout=5.0)
