# search/iterative_deepening.py

from __future__ import annotations
import chess
from .alpha_beta import AlphaBetaSearch, SearchTimeout

INF = 1_000_000


# ==========================================================
# ITERATIVE DEEPENING SEARCH
# ==========================================================

class IterativeDeepeningSearch:

    def __init__(
        self,
        searcher: AlphaBetaSearch,
        time_manager,
        use_aspiration_windows: bool = True,
    ) -> None:
        self.searcher = searcher
        self.time_manager = time_manager
        self.use_aspiration_windows = use_aspiration_windows

        self.last_completed_depth = 0
        self.last_score = 0
        self.pv: list[chess.Move] = []
        self.total_nodes = 0

    # ======================================================
    # MAIN SEARCH ENTRY
    # ======================================================

    def find_best_move(
        self,
        board: chess.Board,
        time_limit: float | None = None,
        max_depth: int = 64,
    ) -> tuple[chess.Move | None, int, int]:
        """
        Main execution framework for managing engine evaluation sequences.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0, 0

        # Safe fallback initialization
        best_move = legal_moves[0]
        best_score = 0
        self.last_completed_depth = 0
        self.total_nodes = 0
        self.pv.clear()

        # ----------------------------------
        # Clock Setup Integration
        # ----------------------------------
        if time_limit is not None:
            allocated = time_limit
        else:
            allocated = self.time_manager.allocate_time(board.turn)

        self.time_manager.start_timer(allocated)

        # Core NNUE stack alignment setup prior to processing branches
        self.searcher.evaluator.nnue.initialize(board)

        # ----------------------------------
        # Iterative Deepening Loop Pass
        # ----------------------------------
        for depth in range(1, max_depth + 1):

            # Check for hard break thresholds if depth limit has been reached
            if self.time_manager.fixed_depth and depth > self.time_manager.max_depth:
                break

            try:
                # --------------------------
                # Aspiration Windows Logic
                # --------------------------
                if self.use_aspiration_windows and depth > 2:
                    # Initialize initial narrow exploration window around last calculated score
                    window = 40
                    alpha = self.last_score - window
                    beta = self.last_score + window

                    score, move = self._search_with_research(board, depth, alpha, beta)
                else:
                    # Baseline full-open evaluation boundary pass
                    score, move = self.searcher.search(board, depth, -INF, INF, is_root=True)

                # Track cumulative nodes explored across processing steps
                self.total_nodes += self.searcher.nodes

                # Save successful, complete iteration passes
                if move is not None:
                    best_move = move
                    best_score = score
                    self.last_score = score
                    self.last_completed_depth = depth

                    self._update_pv(board, move)

                    # Print UCI compliant real-time match diagnostic log string
                    print(f"info depth {depth} score cp {score} nodes {self.total_nodes} time {int(self.time_manager.elapsed() * 1000)} pv {self.pv_string()}")

            except SearchTimeout:
                # Catch time allocations gracefully. Return the previous fully completed search depth tier findings.
                break

            # Mid-tier exit polling checks
            if self.time_manager.should_stop_search():
                break

        return best_move, best_score, self.last_completed_depth

    # ======================================================
    # PROGRESSIVE BOUND RESEARCHING
    # ======================================================

    def _search_with_research(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
    ) -> tuple[int, chess.Move | None]:
        """
        Executes aspiration window searches. Progressively widens bounds
        on fail-low or fail-high anomalies to preserve pruning speeds.
        """
        score, move = self.searcher.search(board, depth, alpha, beta, is_root=True)

        # Fail Low: Position is worse than expected. Open up alpha bound wide.
        if score <= alpha:
            score, move = self.searcher.search(board, depth, -INF, beta, is_root=True)

        # Fail High: Position is significantly better. Open up beta bound wide.
        elif score >= beta:
            score, move = self.searcher.search(board, depth, alpha, INF, is_root=True)

        return score, move

    # ======================================================
    # PRINCIPAL VARIATION DISCOVERY LAYER
    # ======================================================

    def _update_pv(self, board: chess.Board, root_move: chess.Move) -> None:
        """ Reconstructs the primary lookahead PV line using high-speed TT lookups. """
        self.pv.clear()
        temp_board = board.copy()
        move = root_move
        max_length = 64

        while move is not None and len(self.pv) < max_length:
            self.pv.append(move)
            temp_board.push(move)

            try:
                # HIGH-SPEED OPTIMIZATION: Pull native Zobrist keys directly via the internal NNUE cache mapping
                zobrist = self.searcher.evaluator.nnue._hash(temp_board)
                entry = self.searcher.tt.lookup(zobrist)

                if entry is None or entry.best_move is None:
                    break

                # Verify move legality inside temporary board state before stepping deeper
                if entry.best_move in temp_board.legal_moves:
                    move = entry.best_move
                else:
                    break

            except Exception:
                break

    # ======================================================
    # UTILITIES & Telemetry Formatting
    # ======================================================

    def pv_string(self) -> str:
        """ Returns a clean, space-separated string of UCI move characters. """
        return " ".join(move.uci() for move in self.pv)

    def search_info(self) -> dict:
        return {
            "depth": self.last_completed_depth,
            "score": self.last_score,
            "nodes": self.total_nodes,
            "pv": self.pv_string(),
        }

    def reset(self) -> None:
        self.last_completed_depth = 0
        self.last_score = 0
        self.total_nodes = 0
        self.pv.clear()

    def __repr__(self) -> str:
        return f"IterativeDeepeningSearch(depth={self.last_completed_depth}, score={self.last_score})"
