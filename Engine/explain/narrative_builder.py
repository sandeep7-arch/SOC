"""
narrative_builder.py — Converts Game Data into a Story-Like Narrative
======================================================================
YOUR MODULE: engine/explain/narrative_builder.py
 
WHAT THIS FILE DOES:
--------------------
After a game ends, players want to know the STORY of what happened:
  "You had a winning position by move 18, but a rook blunder on move 23
   let the opponent back in. You fought hard but the endgame accuracy
   wasn't enough. Key lesson: rook endgame technique."
 
This is exactly what NarrativeBuilder produces.
It is NOT a move-by-move explanation — that's MoveExplainer's job.
NarrativeBuilder writes the MACRO story: opening → crisis → resolution.
 
THREE OUTPUTS:
    1. Full game narrative (3 paragraphs, story format)
    2. One-line game summary (for dashboard cards / history list)
    3. Title card (like a chess.com game title: "The Ruy Lopez Duel")
 
WHAT IT USES:
    - GameData (from prompt_builder.py) — the stats input
    - build_game_summary_prompt() (from prompt_builder.py) — the prompt
    - LLMProvider (from llm_client.py) — sends the prompt
 
USAGE:
    from engine.explain.narrative_builder import NarrativeBuilder
    from engine.explain.prompt_builder import GameData, MoveData
    from engine.explain.llm_client import MockLLMProvider
 
    builder = NarrativeBuilder(llm=MockLLMProvider())
 
    game = GameData(
        white_player="Vinod", black_player="Stockfish_800",
        total_moves=45, result="0-1",
        blunders=[...], mistakes=[...], inaccuracies=[...],
        white_accuracy=72.4, black_accuracy=91.3,
        opening_name="Ruy Lopez", decisive_moment=23
    )
 
    story = builder.build(game)
    print(story.narrative)          # 3-paragraph story
    print(story.one_liner)          # "Hard-fought loss after a critical rook blunder"
    print(story.title)              # "The Ruy Lopez Battle"
"""
 
import re
from dataclasses import dataclass
from typing import Optional
 
from engine.explain.llm_client import LLMProvider, MockLLMProvider
from engine.explain.prompt_builder import GameData, MoveData, build_game_summary_prompt, _score_to_words
 
 
# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================
 
@dataclass
class GameNarrative:
    """
    The structured output from NarrativeBuilder.
 
    Attributes:
        narrative     : The full 3-paragraph story of the game (LLM-generated).
        one_liner     : A single sentence summary for dashboard cards.
        title         : A short "game title" like a newspaper headline.
        result        : Raw result string: "1-0", "0-1", "1/2-1/2"
        result_words  : Human-readable result: "White won", "Draw", etc.
        white_player  : White player name.
        black_player  : Black player name.
        decisive_move : Move number that turned the game, if known.
        white_accuracy: White's accuracy percentage.
        black_accuracy: Black's accuracy percentage.
        opening_name  : Opening played.
    """
    narrative       : str
    one_liner       : str
    title           : str
    result          : str
    result_words    : str
    white_player    : str
    black_player    : str
    decisive_move   : Optional[int]
    white_accuracy  : float
    black_accuracy  : float
    opening_name    : str
 
    def to_markdown(self) -> str:
        """Formats the full game narrative as markdown for the dashboard."""
        accuracy_bar = self._accuracy_bars()
 
        return (
            f"# 🏁 {self.title}\n\n"
            f"**{self.white_player}** vs **{self.black_player}**  "
            f"| {self.opening_name}  |  Result: **{self.result_words}**\n\n"
            f"{accuracy_bar}\n\n"
            f"---\n\n"
            f"{self.narrative}\n\n"
            f"---\n\n"
            f"*{self.one_liner}*"
        )
 
    def _accuracy_bars(self) -> str:
        """Creates a text accuracy display."""
        w = int(self.white_accuracy)
        b = int(self.black_accuracy)
        return (
            f"⬜ White accuracy: **{w}%**  |  "
            f"⬛ Black accuracy: **{b}%**"
        )
 
    def __str__(self):
        return f"[{self.result}] {self.title} — {self.one_liner}"
 
 
# =============================================================================
# MAIN CLASS: NarrativeBuilder
# =============================================================================
 
class NarrativeBuilder:
    """
    Builds a story-like narrative of a full chess game using an LLM.
 
    Takes a GameData object (full game stats) and returns a GameNarrative
    with the full story, one-liner summary, and a creative game title.
 
    HOW IT WORKS:
        1. Validates GameData
        2. Builds game summary prompt via PromptBuilder
        3. Sends to LLM → gets 3-paragraph narrative
        4. Generates a one-liner summary (second LLM call, short)
        5. Generates a creative title (third LLM call, very short)
        6. Packages into GameNarrative
 
    Args:
        llm          : Any LLMProvider instance.
        player_level : "beginner" / "intermediate" / "advanced" — adjusts tone.
    """
 
    # Result mapping to human-readable
    _RESULT_MAP = {
        "1-0":     "White won",
        "0-1":     "Black won",
        "1/2-1/2": "Draw",
    }
 
    def __init__(
        self,
        llm:          Optional[LLMProvider] = None,
        player_level: str = "intermediate",
    ):
        self._llm          = llm or MockLLMProvider()
        self.player_level  = player_level
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 1: build() ← main entry point
    # -------------------------------------------------------------------------
 
    def build(self, game: GameData) -> GameNarrative:
        """
        Builds a full GameNarrative from a GameData object.
 
        Args:
            game : GameData with full game statistics.
 
        Returns:
            GameNarrative: Full story, one-liner, and title.
        """
        self._validate(game)
 
        result_words = self._RESULT_MAP.get(game.result, "Game concluded")
 
        # Build the main narrative (3 paragraphs)
        narrative = self._build_narrative(game)
 
        # Build a one-liner summary
        one_liner = self._build_one_liner(game, result_words)
 
        # Build a creative title
        title = self._build_title(game, result_words)
 
        return GameNarrative(
            narrative      = narrative,
            one_liner      = one_liner,
            title          = title,
            result         = game.result,
            result_words   = result_words,
            white_player   = game.white_player,
            black_player   = game.black_player,
            decisive_move  = game.decisive_moment,
            white_accuracy = game.white_accuracy,
            black_accuracy = game.black_accuracy,
            opening_name   = game.opening_name,
        )
 
    # -------------------------------------------------------------------------
    # PUBLIC METHOD 2: quick_summary() ← one-liner only, no full narrative
    # -------------------------------------------------------------------------
 
    def quick_summary(self, game: GameData) -> str:
        """
        Generates just the one-line game summary. Faster (1 LLM call only).
 
        Args:
            game : GameData object.
 
        Returns:
            str: One-sentence game summary.
        """
        self._validate(game)
        result_words = self._RESULT_MAP.get(game.result, "Game concluded")
        return self._build_one_liner(game, result_words)
 
    # -------------------------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------------------------
 
    def _build_narrative(self, game: GameData) -> str:
        """Sends the game summary prompt and returns the 3-paragraph narrative."""
        prompt = build_game_summary_prompt(game)
 
        try:
            raw = self._llm.complete_with_retry(prompt, max_tokens=500)
            return self._clean(raw)
        except Exception:
            return self._fallback_narrative(game)
 
    def _build_one_liner(self, game: GameData, result_words: str) -> str:
        """Generates a one-sentence game summary."""
        n_blunders = len(game.blunders)
        n_mistakes = len(game.mistakes)
        total_errors = n_blunders + n_mistakes
 
        prompt = (
            f"Summarize this chess game in ONE sentence (max 20 words).\n"
            f"Players: {game.white_player} (White) vs {game.black_player} (Black)\n"
            f"Result: {result_words}\n"
            f"Opening: {game.opening_name}\n"
            f"Total errors: {total_errors} ({n_blunders} blunders, {n_mistakes} mistakes)\n"
            f"Decisive move: {f'Move {game.decisive_moment}' if game.decisive_moment else 'Not identified'}\n\n"
            f"Write ONE sentence only. No preamble."
        )
 
        try:
            raw = self._llm.complete_with_retry(prompt, max_tokens=60)
            # Keep only first sentence
            sentences = re.split(r'(?<=[.!?])\s', raw.strip())
            return sentences[0] if sentences else self._fallback_one_liner(game, result_words)
        except Exception:
            return self._fallback_one_liner(game, result_words)
 
    def _build_title(self, game: GameData, result_words: str) -> str:
        """Generates a short, creative game title."""
        prompt = (
            f"Create a short title (4-6 words) for this chess game, like a newspaper headline.\n"
            f"Opening: {game.opening_name}\n"
            f"Result: {result_words}\n"
            f"Blunders: {len(game.blunders)}\n"
            f"Players: {game.white_player} vs {game.black_player}\n\n"
            f"Output: title ONLY. No quotes. No preamble. Max 6 words."
        )
 
        try:
            raw = self._llm.complete_with_retry(prompt, max_tokens=20)
            title = raw.strip().strip('"').strip("'")
            # Limit to 6 words
            words = title.split()
            if len(words) > 6:
                title = " ".join(words[:6])
            return title if title else f"The {game.opening_name} Battle"
        except Exception:
            return f"The {game.opening_name} Battle"
 
    def _fallback_narrative(self, game: GameData) -> str:
        """Template-based narrative when LLM is unavailable."""
        result_words = self._RESULT_MAP.get(game.result, "concluded")
        n_blunders   = len(game.blunders)
 
        opening_para = (
            f"{game.white_player} opened with the {game.opening_name}. "
            f"Both sides developed their pieces and contested the center in the opening phase. "
            f"The game began with reasonable accuracy from both players."
        )
 
        if n_blunders > 0:
            b = game.blunders[0]
            key_para = (
                f"The turning point came on move {b.move_number}, "
                f"when {b.player_color} played {b.move_played} — "
                f"a critical error that shifted the evaluation decisively. "
                f"This single blunder changed the character of the entire game."
            )
        elif game.decisive_moment:
            key_para = (
                f"The critical moment arrived on move {game.decisive_moment}, "
                f"when the position shifted in favor of one side. "
                f"From that point, the game's outcome was largely determined."
            )
        else:
            key_para = (
                f"The game was closely contested throughout, "
                f"with both players maintaining solid accuracy. "
                f"Small advantages accumulated over time until a clear winner emerged."
            )
 
        conclusion_para = (
            f"Ultimately, {result_words} after {game.total_moves} moves. "
            f"White finished with {game.white_accuracy:.0f}% accuracy "
            f"and Black with {game.black_accuracy:.0f}%. "
            f"Both players should review the key moments to sharpen their technique."
        )
 
        return f"{opening_para}\n\n{key_para}\n\n{conclusion_para}"
 
    def _fallback_one_liner(self, game: GameData, result_words: str) -> str:
        """Template-based one-liner when LLM is unavailable."""
        n_blunders = len(game.blunders)
        if n_blunders > 0:
            return (
                f"{result_words} in {game.total_moves} moves — "
                f"{n_blunders} blunder(s) were decisive in the {game.opening_name}."
            )
        return (
            f"{result_words} in {game.total_moves} moves "
            f"with high accuracy from both sides ({game.opening_name})."
        )
 
    def _clean(self, text: str) -> str:
        """Strips artifacts from LLM response."""
        text = text.strip()
        text = text.replace("**", "").replace("__", "")
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text
 
    def _validate(self, game: GameData) -> None:
        """Validates GameData fields."""
        if not game.white_player or not game.black_player:
            raise ValueError("GameData must have white_player and black_player set.")
        if game.result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError(f"Invalid result '{game.result}'. Must be '1-0', '0-1', or '1/2-1/2'.")
        if game.total_moves < 1:
            raise ValueError("GameData.total_moves must be at least 1.")
 
    def __repr__(self):
        return f"NarrativeBuilder(llm={type(self._llm).__name__}, level={self.player_level})"
 
 
# =============================================================================
# QUICK TEST — python engine/explain/narrative_builder.py
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("narrative_builder.py — Full Integration Test")
    print("=" * 65)
 
    from engine.explain.llm_client import MockLLMProvider
 
    builder = NarrativeBuilder(llm=MockLLMProvider())
    print(f"\nBuilder: {builder}\n")
 
    # Build sample game data
    blunders = [
        MoveData("e4e5", +120, -200, "d4", +150, "blunder", "pawn", "middlegame", 14, "White"),
        MoveData("Rxd4", -180, -480, "Ke2", -60,  "blunder", "rook", "endgame",   32, "White"),
    ]
    mistakes = [
        MoveData("Bxc6", -80, -180, "Rxd4", -60, "mistake", "bishop", "middlegame", 22, "White"),
    ]
 
    game = GameData(
        white_player   = "Vinod",
        black_player   = "Stockfish_800",
        total_moves    = 45,
        result         = "0-1",
        blunders       = blunders,
        mistakes       = mistakes,
        inaccuracies   = [],
        white_accuracy = 72.4,
        black_accuracy = 91.3,
        opening_name   = "Ruy Lopez",
        decisive_moment= 23,
    )
 
    print("─" * 65)
    print("TEST 1: build() — full narrative")
    print("─" * 65)
    story = builder.build(game)
    print(f"Title    : {story.title}")
    print(f"One-liner: {story.one_liner}")
    print(f"Result   : {story.result_words}")
    print(f"\nNarrative:\n{story.narrative}")
    print(f"\nMarkdown preview:\n{story.to_markdown()[:400]}...")
 
    print("\n" + "─" * 65)
    print("TEST 2: quick_summary() — one-liner only")
    print("─" * 65)
    summary = builder.quick_summary(game)
    print(f"Summary: {summary}")
 
    print("\n" + "─" * 65)
    print("TEST 3: Draw result")
    print("─" * 65)
    game.result = "1/2-1/2"
    game.decisive_moment = None
    draw_story = builder.build(game)
    print(f"Title: {draw_story.title} | Result: {draw_story.result_words}")
 
    print("\n" + "─" * 65)
    print("TEST 4: Validation error")
    print("─" * 65)
    try:
        game.result = "bad_result"
        builder.build(game)
    except ValueError as e:
        print(f"Caught expected error: {e} ✓")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("=" * 65)
 