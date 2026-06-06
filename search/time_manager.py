# search/time_manager.py

from __future__ import annotations
import time

# ==========================================================
# TIME MANAGER
# ==========================================================

class TimeManager:
    """
    Production-grade search time management.
    Handles UCI clocks, sudden death, increments, and custom fixed allocations.
    """

    def __init__(self) -> None:
        # UCI Controls (values in seconds internally for high-speed arithmetic)
        self.wtime = None
        self.btime = None
        self.increment = 0.0
        self.movestogo = None

        # Move allocation targets
        self.allocated_time = 0.0
        self.start_time = None

        # Operating Modes
        self.fixed_depth = False
        self.fixed_time = False
        self.max_depth = 64

        # Diagnostics
        self.stop_requests = 0

    # ======================================================
    # UCI CONTROLS SETUP
    # ======================================================

    def set_time_controls(
        self,
        wtime: int | None,
        btime: int | None,
        increment: int = 0,
        movestogo: int | None = None,
    ) -> None:
        """ Receive incoming clock inputs in milliseconds and convert to seconds. """
        self.wtime = wtime / 1000.0 if wtime is not None else None
        self.btime = btime / 1000.0 if btime is not None else None
        self.increment = increment / 1000.0
        self.movestogo = movestogo

    def set_fixed_depth(self, depth: int) -> None:
        """ Flag engine to search up to a strict depth boundary without time cutoffs. """
        self.fixed_depth = True
        self.fixed_time = False
        self.max_depth = depth

    def set_fixed_move_time(self, milliseconds: int) -> None:
        """ Flag engine to evaluate for an explicit duration window. """
        self.fixed_time = True
        self.fixed_depth = False
        self.allocated_time = milliseconds / 1000.0

    # ======================================================
    # TIME ALLOCATION LOGIC
    # ======================================================

    def allocate_time(self, is_white_to_move: bool) -> float:
        """
        Calculates optimized thinking bounds for the upcoming search window.
        """
        if self.fixed_time:
            return self.allocated_time

        # Fallback if no clock parameters were passed down by GUI
        if self.wtime is None or self.btime is None:
            return 3.0  # Safe default choice (3 seconds)

        remaining = self.wtime if is_white_to_move else self.btime

        # --------------------------------------------------
        # Dynamic Move Divisor Map
        # --------------------------------------------------
        if self.movestogo and self.movestogo > 0:
            # If the GUI gives us a specific control window, use it directly with a small buffer margin
            moves_to_map = self.movestogo + 1
        else:
            # Default progressive game horizon spacing
            moves_to_map = 24.0

        # Calculate base move budget allocation chunk
        base_allocation = remaining / moves_to_map

        # Add a portion of our increment safety buffer
        allocation = base_allocation + (self.increment * 0.75)

        # --------------------------------------------------
        # Hard Security Boundary Clamps
        # --------------------------------------------------
        # Extreme Sudden Death Panic: never spend more than 40% of total remaining time
        hard_ceiling = remaining * 0.40
        if allocation > hard_ceiling:
            allocation = hard_ceiling

        # Absolute Minimum Safety Window (ensure engine has at least 15ms to make a move)
        if allocation < 0.015:
            # If we have an increment, use it as our baseline survival threshold
            allocation = max(0.015, min(self.increment, 0.500))

        self.allocated_time = allocation
        return allocation

    # ======================================================
    # LIFECYCLE MANAGEMENT
    # ======================================================

    def start_timer(self, allocated_time: float) -> None:
        """ Synchronizes start system clock vectors. """
        self.allocated_time = allocated_time
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        """ Returns the time spent in seconds since the search started. """
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    # ======================================================
    # HIGH FREQUENCY SEARCH LOOP INTERRUPT
    # ======================================================

    def should_stop_search(self) -> bool:
        """
        Polled constantly by Alpha-Beta. Highly optimized for speed.
        """
        if self.fixed_depth:
            return False

        if self.start_time is None:
            return False

        # Inlined duration check to eliminate function call overhead inside your tree loop
        if (time.perf_counter() - self.start_time) >= self.allocated_time:
            self.stop_requests += 1
            return True

        return False

    def remaining_time_estimate(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(0.0, self.allocated_time - self.elapsed())

    def reset(self) -> None:
        self.start_time = None
        self.stop_requests = 0

    def stats(self) -> dict:
        return {
            "allocated_time": self.allocated_time,
            "elapsed": self.elapsed(),
            "remaining": self.remaining_time_estimate(),
            "stop_requests": self.stop_requests,
        }

    def __repr__(self) -> str:
        return f"TimeManager(allocated={self.allocated_time:.3f}s, elapsed={self.elapsed():.3f}s)"
