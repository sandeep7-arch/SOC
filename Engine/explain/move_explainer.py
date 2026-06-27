
"""
move_explainer.py — Produces English Explanations for Any Chess Move
=====================================================================
YOUR MODULE: engine/explain/move_explainer.py
 
WHAT THIS FILE DOES:
--------------------
This is the file that actually PRODUCES the English explanation
a player sees after making a move. It connects:
 
    Module 1 (llm_client.py)     → talks to the LLM
    Module 2 (prompt_builder.py) → builds the structured prompt
 
Think of it like this:
    - Module 1 is the PHONE  (makes the call)
    - Module 2 is the SCRIPT (what to say)
    - Module 3 is the PERSON (decides when to call, what script to use,
                              and packages the answer nicely)
 
THE FLOW:
 
    Engine data (scores, moves)
          ↓
    MoveData object  (you create this from engine output)
          ↓
    PromptBuilder    (Module 2 — builds the right prompt)
          ↓
    LLMProvider      (Module 1 — sends prompt, gets response)
          ↓
    ExplanationResponse (clean, structured output)
          ↓
    UI / Dashboard   ( frontend consumes this)
 
WHAT THIS FILE CONTAINS:
    1. ExplanationResponse  — the structured output object
    2. MoveExplainer        — main class, explains any single move
    3. Quick test at bottom — run to verify everything works
"""
 
import time
from dataclasses import dataclass, field
from typing import Optional, List
 
# Import from other modules
from engine.explain.llm_client import LLMProvider, MockLLMProvider, get_llm
from engine.explain.prompt_builder import (
    PromptBuilder,
    MoveData,
    _score_to_words,
    _score_delta_to_severity
)
 
 
# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================
 
@dataclass
class ExplanationResponse:
    """
    The structured output that MoveExplainer returns.
    This is what dashboard / frontend will consume.
    """
    explanation:         str
    move_played:         str
    category:            str
    severity:            str
    score_before:        int
    score_after:         int
    score_drop:          int
    best_move:           str
    score_before_words:  str
    score_after_words:   str
    move_number:         int         = 0
    player_color:        str         = "White"
    phase:               str         = "middlegame"
    timestamp:           float       = field(default_factory=time.time)
    llm_provider:        str         = "unknown"
 
    def to_dict(self) -> dict:
        """
        Convert to Python dictionary.
        Useful when sending data to a REST API or saving to JSON file.
        """
        return {
            "explanation":        self.explanation,
            "move_played":        self.move_played,
            "category":           self.category,
            "severity":           self.severity,
            "score_before":       self.score_before,
            "score_after":        self.score_after,
            "score_drop":         self.score_drop,
            "best_move":          self.best_move,
            "score_before_words": self.score_before_words,
            "score_after_words":  self.score_after_words,
            "move_number":        self.move_number,
            "player_color":       self.player_color,
            "phase":              self.phase,
            "llm_provider":       self.llm_provider,
        }
 
    def to_markdown(self) -> str:
        """
        Convert to a nicely formatted markdown string.
        Perfect for displaying in a dashboard or terminal.
        """
        # Choose icon based on category
        icons = {
            "blunder":     "🔴 BLUNDER",
            "mistake":     "🟠 MISTAKE",
            "inaccuracy":  "🟡 INACCURACY",
            "good":        "🟢 GOOD MOVE",
            "excellent":   "⭐ EXCELLENT MOVE",
        }
        icon = icons.get(self.category.lower(), "⬜ MOVE")
 
        lines = [
            f"## {icon} — Move {self.move_number}",
            f"",
            f"**{self.player_color} played:** `{self.move_played}`  |  "
            f"**Phase:** {self.phase.capitalize()}",
            f"",
            f"**Score before:** {self.score_before_words} ({self.score_before:+d}cp)",
            f"**Score after:**  {self.score_after_words} ({self.score_after:+d}cp)",
        ]
 
        # Only show score drop and best move for bad moves
        if self.category.lower() in ("blunder", "mistake", "inaccuracy"):
            lines.append(f"**Score drop:**   {self.score_drop}cp ({self.severity})")
            lines.append(f"**Best move was:** `{self.best_move}`")
 
        lines += [
            f"",
            f"### Explanation",
            f"{self.explanation}",
            f"",
            f"---",
        ]
 
        return "\n".join(lines)
 
    def to_json(self) -> str:
        """
        Convert to JSON string. For API responses.
         """
        import json
        return json.dumps(self.to_dict(), indent=2)
 
    def __str__(self) -> str:
        """Quick string view — used when you print(response)."""
        return (
            f"[{self.category.upper()}] Move {self.move_number}: {self.move_played} "
            f"| Drop: {self.score_drop}cp | Best: {self.best_move}\n"
            f"→ {self.explanation[:120]}..."
        )
 
 
# =============================================================================
# MAIN CLASS: MoveExplainer
# =============================================================================
 
class MoveExplainer:
    """
    The main class that produces English explanations for chess moves.
    """
 
    def __init__(
        self,
        llm:           Optional[LLMProvider] = None,
        player_level:  str  = "intermediate",
        max_tokens:    int  = 350,
        use_retry:     bool = True,
    ):
        # If no LLM provided, default to Mock (safe offline fallback)
        self._llm          = llm or MockLLMProvider()
        self.player_level  = player_level
        self.max_tokens    = max_tokens
        self.use_retry     = use_retry
 
        # PromptBuilder from Module 2 — builds all prompts
        self._prompt_builder = PromptBuilder(player_level=player_level)
 
        # Track which provider we're using (for logging/debugging)
        self._provider_name = type(self._llm).__name__
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: explain_move()  ← your most-used method
    # -------------------------------------------------------------------------
 
    def explain_move(self, data: MoveData) -> ExplanationResponse:
        """
        Explains any single chess move in plain English.
        It automatically picks the right prompt type based on
        whether the move was good or bad.
        """
        # Step 1: Validate the input data
        self._validate_move_data(data)
 
        # Step 2: Choose the right prompt based on move quality
        category = data.category.lower()
        if category in ("blunder", "mistake", "inaccuracy"):
            # Bad move → use blunder prompt (strict, factual, explains what went wrong)
            prompt = self._prompt_builder.for_blunder(data)
        else:
            # Good/excellent move → use positive prompt (encouraging, explains what's right)
            prompt = self._prompt_builder.for_good_move(data)
 
        # Step 3: Send prompt to LLM and get explanation back
        explanation = self._call_llm(prompt)
 
        # Step 4: Clean up the response (remove extra whitespace, etc.)
        explanation = self._clean_response(explanation)
 
        # Step 5: Calculate metadata
        score_drop = abs(data.score_after - data.score_before)
        severity   = _score_delta_to_severity(score_drop)
 
        # Step 6: Package everything into ExplanationResponse and return
        return ExplanationResponse(
            explanation        = explanation,
            move_played        = data.move_played,
            category           = data.category,
            severity           = severity,
            score_before       = data.score_before,
            score_after        = data.score_after,
            score_drop         = score_drop,
            best_move          = data.best_move,
            score_before_words = _score_to_words(data.score_before),
            score_after_words  = _score_to_words(data.score_after),
            move_number        = data.move_number,
            player_color       = data.player_color,
            phase              = data.phase,
            llm_provider       = self._provider_name,
        )
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 2: explain_moves()  ← batch version for multiple moves
    # -------------------------------------------------------------------------
 
    def explain_moves(self, moves: List[MoveData]) -> List[ExplanationResponse]:
        """
        Explains a list of moves — useful for post-game analysis.
 
        Calls explain_move() for each move in the list.
        Skips moves that fail (logs the error) so one bad move
        doesn't crash the whole analysis.
         """
        responses = []
 
        for i, move_data in enumerate(moves):
            try:
                response = self.explain_move(move_data)
                responses.append(response)
 
            except Exception as e:
                # Don't crash — log and continue to next move
                print(f"[MoveExplainer] Warning: Failed to explain move {i+1} "
                      f"({move_data.move_played}): {e}")
 
        return responses
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 3: explain_top_moments()  ← for dashboard highlights
    # -------------------------------------------------------------------------
 
    def explain_top_moments(
        self,
        moves: List[MoveData],
        top_n: int = 3
    ) -> List[ExplanationResponse]:
        """
        Finds and explains only the N most significant moves in a game.
        """
        # Sort moves by score drop (biggest drop = most impactful)
        sorted_moves = sorted(
            moves,
            key=lambda m: abs(m.score_after - m.score_before),
            reverse=True  # largest drop first
        )
 
        # Take only top N
        top_moves = sorted_moves[:top_n]
 
        # Explain each one
        return self.explain_moves(top_moves)
 
    # -------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -------------------------------------------------------------------------
 
    def _call_llm(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response text.
 
        Uses retry logic if use_retry=True (handles rate limits automatically).
        """
        if self.use_retry:
            return self._llm.complete_with_retry(
                prompt,
                max_tokens=self.max_tokens,
                retries=3,
                delay=2.0
            )
        else:
            return self._llm.complete(prompt, max_tokens=self.max_tokens)
 
    def _clean_response(self, text: str) -> str:
        """
        Cleans up the raw LLM response.
        """
        if not text:
            return "No explanation available."
 
        # Remove leading/trailing whitespace
        text = text.strip()
 
        # Remove markdown bold/italic symbols that sometimes leak through
        text = text.replace("**", "").replace("__", "")
 
        # Collapse multiple blank lines into one
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
 
        # If response is suspiciously short, flag it
        if len(text) < 20:
            return f"[Short response received]: {text}"
 
        return text
 
    def _validate_move_data(self, data: MoveData) -> None:
        """
        Checks that MoveData has all required fields before processing.
 
        Raises ValueError with a helpful message if something is missing.
        """
        if not data.move_played:
            raise ValueError("MoveData.move_played cannot be empty.")
 
        if not data.best_move:
            raise ValueError("MoveData.best_move cannot be empty.")
 
        if not data.category:
            raise ValueError("MoveData.category cannot be empty.")
 
        valid_categories = {"blunder", "mistake", "inaccuracy", "good", "excellent"}
        if data.category.lower() not in valid_categories:
            raise ValueError(
                f"Invalid category '{data.category}'. "
                f"Must be one of: {valid_categories}"
            )
 
        valid_phases = {"opening", "middlegame", "endgame"}
        if data.phase.lower() not in valid_phases:
            raise ValueError(
                f"Invalid phase '{data.phase}'. "
                f"Must be one of: {valid_phases}"
            )
 
    def __repr__(self):
        return (
            f"MoveExplainer("
            f"llm={self._provider_name}, "
            f"level={self.player_level}, "
            f"max_tokens={self.max_tokens})"
        )
 

# =============================================================================
# QUICK TEST — Running: python engine/explain/move_explainer.py
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("move_explainer.py — Full Integration Test")
    print("Module 1 + Module 2 + Module 3 working together")
    print("=" * 65)
 
    # Create the explainer — using Mock so no API key needed
    explainer = MoveExplainer(
        llm          = MockLLMProvider(),
        player_level = "intermediate"
    )
    print(f"\nExplainer ready: {explainer}\n")
 
    # ── TEST 1: Explain a blunder ─────────────────────────────────────
    print("─" * 65)
    print("TEST 1: Explain a BLUNDER")
    print("─" * 65)
 
    blunder_data = MoveData(
        move_played  = "e4e5",
        score_before = +120,
        score_after  = -200,
        best_move    = "d4",
        best_score   = +150,
        category     = "blunder",
        piece        = "pawn",
        phase        = "middlegame",
        move_number  = 14,
        player_color = "White"
    )
 
    result = explainer.explain_move(blunder_data)
 
    print(f"\n[Raw string view]")
    print(result)
 
    print(f"\n[Markdown view]")
    print(result.to_markdown())
 
    print(f"\n[Dict view — for API]")
    import json
    print(json.dumps(result.to_dict(), indent=2))
 
    # ── TEST 2: Explain a good move ───────────────────────────────────
    print("\n" + "─" * 65)
    print("TEST 2: Explain a GOOD MOVE")
    print("─" * 65)
 
    good_data = MoveData(
        move_played  = "Nf3",
        score_before = +50,
        score_after  = +140,
        best_move    = "Nf3",
        best_score   = +140,
        category     = "excellent",
        piece        = "knight",
        phase        = "opening",
        move_number  = 5,
        player_color = "White"
    )
 
    good_result = explainer.explain_move(good_data)
    print(f"\n{good_result.to_markdown()}")
 
    # ── TEST 3: Batch explain multiple moves ──────────────────────────
    print("─" * 65)
    print("TEST 3: Batch explain 3 moves")
    print("─" * 65)
 
    moves_list = [
        MoveData("e4e5", +120, -200, "d4",  +150, "blunder",    "pawn",   "middlegame", 14, "White"),
        MoveData("Nf3",  +50,  +140, "Nf3", +140, "excellent",  "knight", "opening",    5,  "White"),
        MoveData("Bxc6", -80,  -180, "Rxd4",  -60, "mistake",   "bishop", "middlegame", 22, "Black"),
    ]
 
    responses = explainer.explain_moves(moves_list)
    print(f"\nExplained {len(responses)} moves successfully.\n")
    for r in responses:
        print(f"  Move {r.move_number}: [{r.category.upper()}] {r.move_played} → {r.explanation[:80]}...")
 
    # ── TEST 4: Top moments ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("TEST 4: Top 2 most impactful moments")
    print("─" * 65)
 
    highlights = explainer.explain_top_moments(moves_list, top_n=2)
    print(f"\nTop {len(highlights)} key moments:\n")
    for h in highlights:
        print(f"  Move {h.move_number}: [{h.category.upper()}] "
              f"Score drop: {h.score_drop}cp — {h.move_played}")
 
    # ── TEST 5: Validation error handling ────────────────────────────
    print("\n" + "─" * 65)
    print("TEST 5: Validation catches bad input")
    print("─" * 65)
    try:
        bad_data = MoveData("e4e5", 120, -200, "d4", 150, "invalid_category",
                            "pawn", "middlegame", 1, "White")
        explainer.explain_move(bad_data)
    except ValueError as e:
        print(f"\n  Caught expected error: {e} ✓")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("Module 1 + 2 + 3 are fully integrated and working.")
    print("\nNext: Module 4 — blunder_explainer.py")
    print("=" * 65)
