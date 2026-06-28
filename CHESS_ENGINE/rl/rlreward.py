# rl/reward.py

from typing import Optional, Tuple

class RewardEngine:
    """
    Evaluates turn-by-turn equity transitions and flags tactical blunders.
    Outputs raw centipawn metrics to match the K=400.0 NNUEWDLLoss layer.
    """

    def __init__(self, blunder_threshold_cp: float = -150.0) -> None:
        """
        Args:
            blunder_threshold_cp: The centipawn drop indicating a blunder (e.g., -150cp).
        """
        self.blunder_threshold = blunder_threshold_cp

    def calculate_step_reward(
        self,
        current_cp: float,
        previous_cp: Optional[float],
        is_white_to_move: bool
    ) -> Tuple[float, bool]:
        """
        Calculates the centipawn difference caused by a move.

        Returns:
            step_reward: Raw centipawn variance (delta).
            is_blunder: True if the move dropped equity past the safety threshold.
        """
        if previous_cp is None:
            return 0.0, False

        # Calculate the equity delta from the moving player's perspective
        if is_white_to_move:
            delta = current_cp - previous_cp
        else:
            delta = previous_cp - current_cp

        # A blunder occurs if the move drops equity below our threshold
        is_blunder = delta < self.blunder_threshold

        return delta, is_blunder
