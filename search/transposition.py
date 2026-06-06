# search/transposition.py

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Dict, List, Any
import chess


# ==========================================================
# TT FLAGS
# ==========================================================

class TTFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


# ==========================================================
# TT ENTRY
# ==========================================================

@dataclass(slots=True)
class TTEntry:
    """
    Single transposition table entry.
    """
    zobrist: int
    depth: int
    score: int
    flag: TTFlag
    best_move: Optional[chess.Move]


# ==========================================================
# TRANSPOSITION TABLE
# ==========================================================

class TranspositionTable:
    """
    Engine-grade Transposition Table backed by a fixed-size contiguous array structure.

    Replacement Policy:
        Depth Preferred (Deep branch calculations overwrite shallower entry states natively).
    """

    def __init__(self, max_entries: int = 2_000_000):
        """
        Initializes a fixed capacity table cache space.

        Memory Footprint:
            1M entries ≈ 40-50 MB
            2M entries ≈ 80-100 MB
        """
        self.max_entries = max_entries
        # Pre-allocate array slots to completely avoid allocation memory churn during search loops
        self._table: List[Optional[TTEntry]] = [None] * max_entries

        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0

    # ======================================================
    # STORE
    # ======================================================

    def store(
        self,
        zobrist: int,
        depth: int,
        score: int,
        flag: TTFlag,
        best_move: Optional[chess.Move],
        ply: int,
    ) -> None:
        """
        Store an entry into the table using depth-preferred eviction rules.
        """
        index = zobrist % self.max_entries
        existing = self._table[index]

        if existing is not None:
            if existing.zobrist == zobrist:
                # Same position found: Only overwrite if new search depth is deeper
                if depth < existing.depth:
                    return
            else:
                # Collision: Different board configuration maps to the same index array slot
                if depth < existing.depth:
                    return  # Preserve the deeper structural tree path calculation
                self.collisions += 1
        if score > 29000:
            score += ply;
        elif score < -29000:
            score -+ ply
        # Direct in-place array assignment
        self._table[index] = TTEntry(
            zobrist=zobrist,
            depth=depth,
            score=score,
            flag=flag,
            best_move=best_move,
        )
        self.stores += 1

    # ======================================================
    # LOOKUP
    # ======================================================

    def lookup(self, zobrist: int) -> Optional[TTEntry]:
        """
        Lookup raw entry state matching target position signature hash.
        """
        index = zobrist % self.max_entries
        entry = self._table[index]

        if entry is None or entry.zobrist != zobrist:
            self.misses += 1
            return None

        self.hits += 1
        return entry

    # ======================================================
    # PROBE (ALPHA-BETA INTEGRATION)
    # ======================================================

    def probe(
        self,
        zobrist: int,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
    ) -> Optional[int]:
        """
        Probe table records to evaluate deep branch cutoff eligibility.
        """
        index = zobrist % self.max_entries
        entry = self._table[index]

        # Inlined lookup validation to maximize processing cycle speeds
        if entry is None or entry.zobrist != zobrist:
            self.misses += 1
            return None

        self.hits += 1

        # Deep calculations are required to satisfy shallow node evaluations safely
        if entry.depth < depth:
            return None

        if entry.score > 29000:
            entry.score -= ply
        elif entry.score < -29000:
            entry.score += ply

        if entry.flag == TTFlag.EXACT:
            return entry.score

        if entry.flag == TTFlag.LOWERBOUND:
            if entry.score >= beta:
                return entry.score

        elif entry.flag == TTFlag.UPPERBOUND:
            if entry.score <= alpha:
                return entry.score

        return None

    # ======================================================
    # CLEAN BEST MOVE LOOKUP (FIXED METRIC POLLUTION)
    # ======================================================

    def best_move(self, zobrist: int) -> Optional[chess.Move]:
        """
        Extract move recommendations for search ordering without corrupting diagnostic counters.
        """
        index = zobrist % self.max_entries
        entry = self._table[index]

        if entry is not None and entry.zobrist == zobrist:
            return entry.best_move
        return None

    # ======================================================
    # UTILITIES & DIAGNOSTICS
    # ======================================================

    def clear(self) -> None:
        """Flush active allocations back to clean slate state."""
        for i in range(self.max_entries):
            self._table[i] = None

        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0

    def size(self) -> int:
        """Count active occupied element positions inside the table array."""
        return sum(1 for entry in self._table if entry is not None)

    def capacity(self) -> int:
        return self.max_entries

    def load_factor(self) -> float:
        return self.size() / self.max_entries

    def stats(self) -> dict[str, Any]:
        """Compile internal usage statistics."""
        total_lookups = self.hits + self.misses
        hit_rate = self.hits / total_lookups if total_lookups > 0 else 0.0

        return {
            "entries": self.size(),
            "capacity": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "collisions": self.collisions,
            "stores": self.stores,
        }

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, zobrist: int) -> bool:
        index = zobrist % self.max_entries
        entry = self._table[index]
        return entry is not None and entry.zobrist == zobrist

    def __repr__(self) -> str:
        return f"TranspositionTable(capacity={self.max_entries})"
