"""Small Elo helper for promotion gates and offline arena summaries."""

from __future__ import annotations

import math


class EloTracker:
    def __init__(self, initial_champion_elo: float = 1500.0, k_factor: float = 32.0) -> None:
        self.champion_elo = initial_champion_elo
        self.candidate_elo = initial_champion_elo
        self.k_factor = k_factor

    @staticmethod
    def calculate_expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_ratings(self, candidate_score: float) -> tuple[float, float]:
        """Update ratings from a score in [0, 1] from the candidate perspective."""
        candidate_score = max(0.0, min(1.0, candidate_score))
        expected_candidate = self.calculate_expected_score(self.candidate_elo, self.champion_elo)
        expected_champion = 1.0 - expected_candidate
        self.candidate_elo += self.k_factor * (candidate_score - expected_candidate)
        self.champion_elo += self.k_factor * ((1.0 - candidate_score) - expected_champion)
        return self.candidate_elo, self.champion_elo

    def passed_gate(self, wins: int, draws: int, losses: int) -> bool:
        games = wins + draws + losses
        if games <= 0:
            return False
        score = (wins + 0.5 * draws) / games
        self.update_ratings(score)
        return score > 0.5
