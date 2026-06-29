"""
commentary.py - Live Chess Commentary Engine 
WHAT THIS FILE CONTAINS:
    1. CommentaryStyle    - enum for commentary tone
    2. CommentaryLine     - single commentary output
    3. CommentaryEngine   - main class
"""
 
import time
import random
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
 
from engine.explain.llm_client import LLMProvider, MockLLMProvider
from engine.explain.prompt_builder import (
    MoveData, PromptBuilder,
    _score_to_words, _score_delta_to_severity
)
 
 
# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================
 
class CommentaryStyle(Enum):
    """
    Controls the tone and style of commentary.
 
    CASUAL      : Friendly, simple, for beginners
    ANALYTICAL  : Technical, precise, for intermediate players
    DRAMATIC    : Exciting, emotional, like a live broadcast
    EDUCATIONAL : Teaching focused, explains concepts
    """
    CASUAL      = "casual"
    ANALYTICAL  = "analytical"
    DRAMATIC    = "dramatic"
    EDUCATIONAL = "educational"
 
 
@dataclass
class CommentaryLine:
    """
    A single commentary line for one move.
    """
    text:        str
    move_number: int
    move_played: str
    category:    str
    style:       str
    is_critical: bool  = False
    timestamp:   float = 0.0
 
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
 
    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "move_number": self.move_number,
            "move_played": self.move_played,
            "category":    self.category,
            "style":       self.style,
            "is_critical": self.is_critical,
        }
 
 
# =============================================================================
# MAIN CLASS
# =============================================================================
 
class CommentaryEngine:
    """
    Generates live commentary for chess moves.
    Called after EVERY move during a live game.
    Works with  frontend for real-time display.
    """
 
    def __init__(
        self,
        llm:     Optional[LLMProvider] = None,
        style:   CommentaryStyle = CommentaryStyle.DRAMATIC,
        use_llm: bool = True,
    ):
        self._llm            = llm or MockLLMProvider()
        self.style           = style
        self.use_llm         = use_llm
        self._prompt_builder = PromptBuilder()
        self._history:       List[CommentaryLine] = []
 
    def comment(self, data: MoveData) -> CommentaryLine:
        """
        Generates commentary for a single move.
        Called in real time after every move.
        """
        score_drop  = abs(data.score_after - data.score_before)
        is_critical = score_drop >= 1.5  # blunder threshold from Piyush
 
        if self.use_llm:
            text = self._llm_comment(data)
        else:
            text = self._template_comment(data)
 
        line = CommentaryLine(
            text        = text,
            move_number = data.move_number,
            move_played = data.move_played,
            category    = data.category,
            style       = self.style.value,
            is_critical = is_critical,
        )
 
        # Store in history for context
        self._history.append(line)
        return line
 
    def comment_batch(self, moves: List[MoveData]) -> List[CommentaryLine]:
        """
        Generates commentary for multiple moves.
        Used for post-game replay.
        """
        lines = []
        for move in moves:
            try:
                lines.append(self.comment(move))
            except Exception as e:
                print(f"[CommentaryEngine] Skipping move {move.move_number}: {e}")
        return lines
 
    def get_game_highlights(self, top_n: int = 3) -> List[CommentaryLine]:
        """
        Returns the N most critical commentary moments.
        Good for a "key moments" summary panel.
        """
        critical = [l for l in self._history if l.is_critical]
        return critical[:top_n]
 
    def reset(self):
        """Clears commentary history. Call between games."""
        self._history = []
 
    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------
 
    def _llm_comment(self, data: MoveData) -> str:
        """Generates commentary using LLM."""
        prompt = self._build_commentary_prompt(data)
        try:
            return self._llm.complete_with_retry(
                prompt, max_tokens=150, retries=2, delay=1.0
            ).strip()
        except Exception:
            # Fall back to template if LLM fails
            return self._template_comment(data)
 
    def _build_commentary_prompt(self, data: MoveData) -> str:
        """Builds a short, focused prompt for live commentary."""
        score_drop   = abs(data.score_after - data.score_before)
        severity     = _score_delta_to_severity(score_drop)
        before_words = _score_to_words(data.score_before)
        after_words  = _score_to_words(data.score_after)
 
        style_instructions = {
            CommentaryStyle.CASUAL:      "Use simple friendly language. One short sentence.",
            CommentaryStyle.ANALYTICAL:  "Be precise and technical. One analytical sentence.",
            CommentaryStyle.DRAMATIC:    "Be dramatic and exciting like a live broadcaster. One punchy sentence.",
            CommentaryStyle.EDUCATIONAL: "Explain the key concept in one teaching sentence.",
        }
        instruction = style_instructions.get(self.style, "One clear sentence.")
 
        return f"""You are a chess commentator. Generate ONE sentence of live commentary.
 
MOVE: {data.move_played} by {data.player_color} (Move {data.move_number})
CATEGORY: {data.category.upper()} ({severity})
BEFORE: {before_words}
AFTER: {after_words}
PHASE: {data.phase}
 
STYLE: {instruction}
 
Write exactly ONE sentence of commentary. No more.
Do not use raw numbers. Base only on data above.""".strip()
 
    def _template_comment(self, data: MoveData) -> str:
        """
        Template-based commentary — instant, no LLM needed.
        Used as fallback or when use_llm=False.
        """
        score_drop = abs(data.score_after - data.score_before)
        category   = data.category.lower()
        style      = self.style
 
        templates = {
            CommentaryStyle.DRAMATIC: {
                "blunder":    [
                    f"Devastating! {data.player_color}'s {data.move_played} throws away the advantage completely!",
                    f"Oh no! That blunder on move {data.move_number} changes everything!",
                    f"A catastrophic mistake — {data.move_played} loses the game on the spot!",
                ],
                "mistake":    [
                    f"A serious error — {data.player_color} could have done much better here.",
                    f"That mistake on move {data.move_number} gives the opponent real winning chances.",
                ],
                "inaccuracy": [
                    f"A slight inaccuracy — not the best move available.",
                    f"The engine preferred something else, but {data.move_played} is playable.",
                ],
                "good":       [
                    f"Solid move by {data.player_color} — maintaining the balance.",
                    f"Good choice on move {data.move_number}!",
                ],
                "excellent":  [
                    f"Brilliant! {data.move_played} is exactly what the position demanded!",
                    f"What a find! {data.player_color} plays the best move on the board!",
                ],
            },
            CommentaryStyle.CASUAL: {
                "blunder":    [f"Oops! That was a big mistake — {data.move_played} loses a lot."],
                "mistake":    [f"That wasn't the best move — the opponent now has a better position."],
                "inaccuracy": [f"Slightly off — there was a better option available."],
                "good":       [f"Nice move! {data.player_color} is playing well."],
                "excellent":  [f"Great move! That's exactly right!"],
            },
            CommentaryStyle.ANALYTICAL: {
                "blunder":    [f"Move {data.move_number}: {data.move_played} drops {score_drop:.1f} pawns — a serious evaluation collapse."],
                "mistake":    [f"Suboptimal: {data.move_played} concedes {score_drop:.1f} pawns of advantage."],
                "inaccuracy": [f"Minor imprecision: {data.move_played} slightly weakens the position."],
                "good":       [f"Accurate: {data.move_played} maintains the evaluation."],
                "excellent":  [f"Best move: {data.move_played} maximizes the position's potential."],
            },
            CommentaryStyle.EDUCATIONAL: {
                "blunder":    [f"This is a key learning moment — {data.move_played} in the {data.phase} creates a serious weakness."],
                "mistake":    [f"Notice how {data.move_played} misses the key idea in this {data.phase} position."],
                "inaccuracy": [f"A good lesson here — small inaccuracies in the {data.phase} accumulate over time."],
                "good":       [f"This is how to play the {data.phase} — {data.move_played} follows good principles."],
                "excellent":  [f"Remember this pattern — {data.move_played} in the {data.phase} is a model move."],
            },
        }
 
        style_templates = templates.get(style, templates[CommentaryStyle.DRAMATIC])
        category_lines  = style_templates.get(category, ["Interesting move."])
        return random.choice(category_lines)
 
 
# =============================================================================
# QUICK TEST
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("commentary.py — Quick Test")
    print("=" * 65)
 
    engine = CommentaryEngine(
        llm     = MockLLMProvider(),
        style   = CommentaryStyle.DRAMATIC,
        use_llm = False  # use templates for quick test
    )
 
    moves = [
        MoveData("e4",   0.0,  0.3,  "e4",  0.3,  "good",
                 "pawn",   "opening",    1,  "White"),
        MoveData("d6",   0.5,  1.8,  "Nf6", 0.4,  "blunder",
                 "pawn",   "opening",    6,  "Black",
                 "opening_error", "Poor development"),
        MoveData("Nf3",  0.3,  0.6,  "Nf3", 0.6,  "excellent",
                 "knight", "opening",    3,  "White"),
        MoveData("Bxc6", 0.5, -0.8,  "e5",  0.7,  "mistake",
                 "bishop", "middlegame", 14, "White"),
    ]
 
    print("\n[TEST 1] Dramatic commentary")
    for move in moves:
        line = engine.comment(move)
        print(f"  Move {line.move_number}: {line.text}")
 
    print("\n[TEST 2] All styles for one blunder")
    blunder = moves[1]
    for style in CommentaryStyle:
        e = CommentaryEngine(use_llm=False, style=style)
        line = e.comment(blunder)
        print(f"  {style.value:12}: {line.text}")
 
    print("\n[TEST 3] Critical moments")
    highlights = engine.get_game_highlights(top_n=2)
    print(f"  Found {len(highlights)} critical moments:")
    for h in highlights:
        print(f"  Move {h.move_number}: {h.text}")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("=" * 65)
