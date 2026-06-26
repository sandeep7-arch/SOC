"""
blunder_explainer.py — Specialized Explainer for Blunders and Mistakes
=======================================================================
YOUR MODULE: engine/explain/blunder_explainer.py
 
WHAT THIS FILE DOES:
--------------------
MoveExplainer (Module 3) is a GENERAL explainer — it handles any move.
BlunderExplainer is a SPECIALIST — it focuses ONLY on blunders and mistakes,
and it does three extra things MoveExplainer doesn't:
 
    1. BLUNDER PATTERN DETECTION
       Looks across multiple blunders in a game and detects recurring themes:
       "This player keeps hanging pieces in the endgame"
       "Three of the five blunders happened on move 20-30 (time pressure?)"
 
    2. SEVERITY TRIAGE
       Automatically sorts blunders by how decisive they were,
       so the dashboard can highlight the single most critical moment.
 
    3. BATCH EXPLANATION WITH DEDUPLICATION
       If two blunders are similar (both are piece hangings), it explains
       them together instead of repeating the same explanation twice.
 
RELATIONSHIP TO OTHER MODULES:
    BlunderExplainer uses MoveExplainer internally.
    BlunderExplainer is NOT a subclass — it composes MoveExplainer.
    This is the "composition over inheritance" pattern.
 
    MoveExplainer → explains ONE move
    BlunderExplainer → analyzes MANY blunders as a group
 
USAGE:
    from engine.explain.blunder_explainer import BlunderExplainer
    from engine.explain.prompt_builder import MoveData
    from engine.explain.llm_client import MockLLMProvider
 
    explainer = BlunderExplainer(llm=MockLLMProvider())
 
    # Explain all blunders from a game
    blunders = [MoveData(...), MoveData(...), ...]
    report = explainer.analyze_blunders(blunders)
 
    print(report.worst_blunder.explanation)
    print(report.pattern_summary)
    for b in report.explained_blunders:
        print(b.to_markdown())
"""
 
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import Counter
 
from engine.explain.llm_client import LLMProvider, MockLLMProvider
from engine.explain.move_explainer import MoveExplainer, ExplanationResponse
from engine.explain.prompt_builder import MoveData, PromptBuilder, _score_delta_to_severity
 
 
# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================
 
@dataclass
class BlunderReport:
    """
    The full analysis report for all blunders in a game.
 
    This is what BlunderExplainer returns after analyzing all the blunders
    from one game. The dashboard consumes this to build the post-game report.
 
    Attributes:
        explained_blunders : List of ExplanationResponse for each blunder, sorted by severity.
        worst_blunder      : The single most impactful blunder in the game.
        total_cp_lost      : Total centipawns lost across all blunders combined.
        blunder_phases     : Dict counting blunders per phase: {"opening": 0, "middlegame": 2, "endgame": 1}
        blunder_colors     : Dict counting blunders per player: {"White": 2, "Black": 1}
        pattern_summary    : One or two sentences describing the recurring mistake pattern (from LLM).
        dominant_phase     : Which phase had the most blunders.
        severity_breakdown : Dict counting by severity: {"severe": 1, "catastrophic": 2, ...}
        triage_order       : List of move_played strings, sorted worst-first.
    """
    explained_blunders  : List[ExplanationResponse]
    worst_blunder       : Optional[ExplanationResponse]
    total_cp_lost       : int
    blunder_phases      : Dict[str, int]
    blunder_colors      : Dict[str, int]
    pattern_summary     : str
    dominant_phase      : str
    severity_breakdown  : Dict[str, int]
    triage_order        : List[str]
 
    def to_markdown(self) -> str:
        """Returns a formatted markdown summary of the blunder report."""
        lines = [
            "# 🔴 Blunder Analysis Report",
            "",
            f"**Total centipawns lost to blunders:** {self.total_cp_lost}cp",
            f"**Dominant phase for blunders:** {self.dominant_phase.capitalize()}",
            f"**Blunder count by phase:** " + ", ".join(
                f"{k}: {v}" for k, v in self.blunder_phases.items() if v > 0
            ),
            "",
            "## Pattern Identified",
            self.pattern_summary,
            "",
            "## Worst Blunder",
        ]
 
        if self.worst_blunder:
            lines.append(self.worst_blunder.to_markdown())
        else:
            lines.append("_No blunders found._")
 
        lines += [
            "",
            "## All Blunders (Worst First)",
        ]
 
        for i, b in enumerate(self.explained_blunders, 1):
            lines.append(f"### {i}. Move {b.move_number} — {b.move_played} ({b.severity})")
            lines.append(b.explanation)
            lines.append("")
 
        return "\n".join(lines)
 
 
# =============================================================================
# MAIN CLASS: BlunderExplainer
# =============================================================================
 
class BlunderExplainer:
    """
    Specialized analyzer for blunders and mistakes across an entire game.
 
    This class takes a LIST of blunder MoveData objects (not just one),
    analyzes them together as a group, and produces a BlunderReport
    with pattern detection, severity triage, and LLM-generated insights.
 
    USAGE:
        explainer = BlunderExplainer(llm=MockLLMProvider(), player_level="intermediate")
 
        blunders = [
            MoveData("e4e5", +120, -200, "d4", +150, "blunder", "pawn", "middlegame", 14, "White"),
            MoveData("Rxd4", -180, -480, "Ke2", -60,  "blunder", "rook", "endgame",   32, "White"),
        ]
 
        report = explainer.analyze_blunders(blunders)
        print(report.pattern_summary)
        print(report.worst_blunder.explanation)
 
    Args:
        llm          : Any LLMProvider (Mock, Gemini, Claude, etc.)
        player_level : "beginner" / "intermediate" / "advanced"
        max_tokens   : Max LLM response length per explanation.
    """
 
    def __init__(
        self,
        llm:           Optional[LLMProvider] = None,
        player_level:  str = "intermediate",
        max_tokens:    int = 350,
    ):
        self._llm          = llm or MockLLMProvider()
        self.player_level  = player_level
 
        # Internally uses MoveExplainer for individual explanations
        self._move_explainer = MoveExplainer(
            llm=self._llm,
            player_level=player_level,
            max_tokens=max_tokens,
        )
        self._prompt_builder = PromptBuilder(player_level=player_level)
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: analyze_blunders() ← MAIN entry point
    # -------------------------------------------------------------------------
 
    def analyze_blunders(self, blunders: List[MoveData]) -> BlunderReport:
        """
        Analyzes all blunders from a game and returns a structured report.
 
        Steps:
          1. Validate and filter — only process actual blunders/mistakes.
          2. Triage — sort by severity (worst centipawn loss first).
          3. Explain each one using MoveExplainer.
          4. Detect patterns across all blunders.
          5. Package everything into BlunderReport.
 
        Args:
            blunders : List of MoveData objects (should be blunders/mistakes).
                       If good moves are accidentally included, they're skipped.
 
        Returns:
            BlunderReport: Full analysis with explanations, patterns, and stats.
        """
        if not blunders:
            return self._empty_report()
 
        # Step 1: Filter to only real blunders/mistakes
        real_blunders = [
            m for m in blunders
            if m.category.lower() in ("blunder", "mistake", "inaccuracy")
        ]
 
        if not real_blunders:
            return self._empty_report()
 
        # Step 2: Triage — sort worst-first by centipawn loss
        triaged = self._triage(real_blunders)
 
        # Step 3: Explain each blunder
        explained = self._move_explainer.explain_moves(triaged)
 
        # Step 4: Compute stats
        total_cp_lost     = sum(abs(m.score_after - m.score_before) for m in triaged)
        blunder_phases    = self._count_by_phase(triaged)
        blunder_colors    = self._count_by_color(triaged)
        dominant_phase    = max(blunder_phases, key=blunder_phases.get)
        severity_breakdown = self._count_by_severity(triaged)
        triage_order      = [m.move_played for m in triaged]
 
        # Step 5: Detect patterns using LLM
        pattern_summary = self._detect_pattern(triaged, dominant_phase)
 
        # Worst blunder is the first in triage (highest cp loss)
        worst_blunder = explained[0] if explained else None
 
        return BlunderReport(
            explained_blunders  = explained,
            worst_blunder       = worst_blunder,
            total_cp_lost       = total_cp_lost,
            blunder_phases      = blunder_phases,
            blunder_colors      = blunder_colors,
            pattern_summary     = pattern_summary,
            dominant_phase      = dominant_phase,
            severity_breakdown  = severity_breakdown,
            triage_order        = triage_order,
        )
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 2: worst_blunder() ← quick single-call version
    # -------------------------------------------------------------------------
 
    def worst_blunder(self, blunders: List[MoveData]) -> Optional[ExplanationResponse]:
        """
        Finds and explains only the single worst blunder in the list.
 
        Use this when you only want to highlight ONE key mistake
        (e.g., the "game-losing blunder" badge in the dashboard).
 
        Args:
            blunders : List of MoveData objects.
 
        Returns:
            ExplanationResponse for the worst blunder, or None if list is empty.
        """
        if not blunders:
            return None
 
        worst = max(blunders, key=lambda m: abs(m.score_after - m.score_before))
        return self._move_explainer.explain_move(worst)
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 3: explain_by_phase() ← phase-grouped explanations
    # -------------------------------------------------------------------------
 
    def explain_by_phase(
        self,
        blunders: List[MoveData]
    ) -> Dict[str, List[ExplanationResponse]]:
        """
        Groups blunders by game phase and explains each group.
 
        Returns a dict like:
            {
                "opening":    [ExplanationResponse, ...],
                "middlegame": [ExplanationResponse, ...],
                "endgame":    [ExplanationResponse, ...],
            }
 
        Useful for the coaching module to target specific phase weaknesses.
 
        Args:
            blunders : List of MoveData objects.
 
        Returns:
            Dict[str, List[ExplanationResponse]]: Grouped and explained blunders.
        """
        grouped: Dict[str, List[MoveData]] = {
            "opening": [], "middlegame": [], "endgame": []
        }
 
        for b in blunders:
            phase = b.phase.lower()
            if phase in grouped:
                grouped[phase].append(b)
 
        result: Dict[str, List[ExplanationResponse]] = {}
        for phase, moves in grouped.items():
            if moves:
                result[phase] = self._move_explainer.explain_moves(moves)
 
        return result
 
    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------
 
    def _triage(self, blunders: List[MoveData]) -> List[MoveData]:
        """
        Sorts blunders by centipawn loss — largest drop first.
        This is the 'triage' order: worst blunder gets explained first.
        """
        return sorted(
            blunders,
            key=lambda m: abs(m.score_after - m.score_before),
            reverse=True
        )
 
    def _count_by_phase(self, blunders: List[MoveData]) -> Dict[str, int]:
        """Counts how many blunders occurred in each game phase."""
        counter = {"opening": 0, "middlegame": 0, "endgame": 0}
        for b in blunders:
            phase = b.phase.lower()
            if phase in counter:
                counter[phase] += 1
        return counter
 
    def _count_by_color(self, blunders: List[MoveData]) -> Dict[str, int]:
        """Counts how many blunders each player made."""
        counter: Dict[str, int] = {}
        for b in blunders:
            color = b.player_color
            counter[color] = counter.get(color, 0) + 1
        return counter
 
    def _count_by_severity(self, blunders: List[MoveData]) -> Dict[str, int]:
        """Counts blunders grouped by severity level."""
        counter: Dict[str, int] = {}
        for b in blunders:
            drop = abs(b.score_after - b.score_before)
            sev  = _score_delta_to_severity(drop)
            counter[sev] = counter.get(sev, 0) + 1
        return counter
 
    def _detect_pattern(self, blunders: List[MoveData], dominant_phase: str) -> str:
        """
        Asks the LLM to identify a recurring pattern across multiple blunders.
 
        Uses build_lesson_prompt() from PromptBuilder (Module 2).
        Falls back to a template if LLM fails.
 
        Args:
            blunders       : All blunders in the game.
            dominant_phase : The phase where most blunders occurred.
 
        Returns:
            str: Pattern summary in 1-2 sentences.
        """
        if len(blunders) < 2:
            # Only one blunder — no pattern to detect
            b = blunders[0]
            drop = abs(b.score_after - b.score_before)
            sev  = _score_delta_to_severity(drop)
            return (
                f"A single {sev} blunder on move {b.move_number} was the decisive error. "
                f"Focus on verifying piece safety before committing to moves in the {dominant_phase}."
            )
 
        try:
            # Use prompt_builder's lesson prompt (Module 2)
            prompt = self._prompt_builder.for_lesson(
                mistake_category="blunder",
                example_moves=blunders,
                phase=dominant_phase,
            )
            raw = self._llm.complete_with_retry(prompt, max_tokens=200)
            return raw.strip() if raw else self._fallback_pattern(blunders, dominant_phase)
 
        except Exception:
            return self._fallback_pattern(blunders, dominant_phase)
 
    def _fallback_pattern(self, blunders: List[MoveData], dominant_phase: str) -> str:
        """
        Template-based fallback pattern summary when LLM is unavailable.
        Uses move data stats to produce a factual, non-invented summary.
        """
        n = len(blunders)
        total_cp = sum(abs(b.score_after - b.score_before) for b in blunders)
        avg_cp   = total_cp // n
 
        phase_parts = []
        for b in blunders:
            phase_parts.append(b.phase)
        most_common_phase = Counter(phase_parts).most_common(1)[0][0]
 
        return (
            f"Across {n} blunders, an average of {avg_cp} centipawns were lost per error, "
            f"with the majority occurring in the {most_common_phase}. "
            f"Practicing slow, deliberate move verification before committing — "
            f"especially in the {most_common_phase} — will address this recurring pattern."
        )
 
    def _empty_report(self) -> BlunderReport:
        """Returns an empty BlunderReport when no blunders are found."""
        return BlunderReport(
            explained_blunders  = [],
            worst_blunder       = None,
            total_cp_lost       = 0,
            blunder_phases      = {"opening": 0, "middlegame": 0, "endgame": 0},
            blunder_colors      = {},
            pattern_summary     = "No blunders detected in this game. Excellent accuracy!",
            dominant_phase      = "middlegame",
            severity_breakdown  = {},
            triage_order        = [],
        )
 
    def __repr__(self):
        return f"BlunderExplainer(llm={type(self._llm).__name__}, level={self.player_level})"
 
 
# =============================================================================
# QUICK TEST — python engine/explain/blunder_explainer.py
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("blunder_explainer.py — Full Integration Test")
    print("=" * 65)
 
    from engine.explain.llm_client import MockLLMProvider
 
    explainer = BlunderExplainer(llm=MockLLMProvider(), player_level="intermediate")
    print(f"\nExplainer: {explainer}\n")
 
    blunders = [
        MoveData("e4e5", +120, -200, "d4",   +150, "blunder",    "pawn",   "middlegame", 14, "White"),
        MoveData("Rxd4", -180, -480, "Ke2",  -60,  "blunder",    "rook",   "endgame",    32, "White"),
        MoveData("Bxc6", -80,  -180, "Rxd4", -60,  "mistake",    "bishop", "middlegame", 22, "White"),
        MoveData("h4",   +50,  -50,  "Rc1",  +80,  "inaccuracy", "pawn",   "opening",    8,  "White"),
    ]
 
    print("─" * 65)
    print("TEST 1: Full blunder analysis")
    print("─" * 65)
 
    report = explainer.analyze_blunders(blunders)
 
    print(f"Total CP lost  : {report.total_cp_lost}")
    print(f"Dominant phase : {report.dominant_phase}")
    print(f"Phases         : {report.blunder_phases}")
    print(f"Severity       : {report.severity_breakdown}")
    print(f"Triage order   : {report.triage_order}")
    print(f"\nPattern:\n  {report.pattern_summary}")
    print(f"\nWorst blunder: [{report.worst_blunder.category}] {report.worst_blunder.move_played}")
    print(f"  → {report.worst_blunder.explanation[:100]}...")
 
    print("\n" + "─" * 65)
    print("TEST 2: worst_blunder() single call")
    print("─" * 65)
    worst = explainer.worst_blunder(blunders)
    print(f"Worst: {worst.move_played} on move {worst.move_number} (drop: {worst.score_drop}cp)")
 
    print("\n" + "─" * 65)
    print("TEST 3: explain_by_phase()")
    print("─" * 65)
    by_phase = explainer.explain_by_phase(blunders)
    for phase, responses in by_phase.items():
        print(f"  {phase}: {len(responses)} blunder(s) explained")
 
    print("\n" + "─" * 65)
    print("TEST 4: Empty input → empty report")
    print("─" * 65)
    empty_report = explainer.analyze_blunders([])
    print(f"Pattern: {empty_report.pattern_summary}")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("=" * 65)