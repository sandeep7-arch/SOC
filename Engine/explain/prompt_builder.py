"""
prompt_builder.py — Chess Engine Data → Structured LLM Prompts
===============================================================
YOUR MODULE: engine/explain/prompt_builder.py
 
WHY THIS FILE IS THE MOST IMPORTANT:
-------------------------------------
If you just ask an LLM "why was Nf3 a blunder?", it will MAKE UP an answer.
It might say "because it weakens the kingside" even if that's completely wrong.
This is called hallucination — and it's the #1 enemy of your project.
 
THE SOLUTION — Ground the LLM in engine data:
    BAD  prompt: "Why was Nf3 a blunder?"
    GOOD prompt: "Move: Nf3. Score before: +120cp. Score after: -200cp.
                  Best move was: d4. Explain ONLY using this data."
 
This file builds those GOOD prompts. Every single fact the LLM gets
must come from the engine. Nothing invented. Nothing assumed.
 
THE CORE INSIGHT:
    You are NOT asking the LLM to reason about chess.
    You are asking the LLM to translate numbers into English sentences.
    The chess reasoning is done by the engine. The LLM just explains it.
 
WHAT THIS FILE CONTAINS:
    1. _score_to_words()          — converts +120 → "White has a clear advantage"
    2. _score_delta_to_severity() — converts -320 → "catastrophic"
    3. _get_phase_context()       — opening/middlegame/endgame tips
    4. build_blunder_prompt()     — main prompt for a blunder/mistake
    5. build_good_move_prompt()   — prompt for explaining a good move
    6. build_game_summary_prompt()— prompt for full game narrative
    7. build_coaching_prompt()    — prompt for interactive coaching mode
    8. PromptBuilder class        — wraps all the above neatly
"""
 
from dataclasses import dataclass
from typing import Optional, List
 
 
# =============================================================================
# DATA STRUCTURES
# =============================================================================
 
@dataclass
class MoveData:
    """
    All the engine data about a single move.
    This is what the engine gives you. You feed this into the prompt builder.
 
    Attributes:
        move_played  : The move in UCI notation e.g. "e2e4" or SAN e.g. "Nf3"
        score_before : Engine score BEFORE this move in centipawns (+ve = White winning)
        score_after  : Engine score AFTER this move in centipawns
        best_move    : The move the engine recommended instead
        best_score   : Engine score if best_move had been played
        category     : "blunder" / "mistake" / "inaccuracy" / "good" / "excellent"
        piece        : Which piece moved e.g. "knight", "pawn", "queen"
        phase        : "opening" / "middlegame" / "endgame"
        move_number  : What move number in the game (e.g. 15)
        player_color : "White" or "Black"
        fen_before   : Board position in FEN notation before the move (optional)
 
    Example (you create this from engine output):
        data = MoveData(
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
    """
    move_played:   str
    score_before:  int
    score_after:   int
    best_move:     str
    best_score:    int
    category:      str
    piece:         str         = "piece"
    phase:         str         = "middlegame"
    move_number:   int         = 0
    player_color:  str         = "White"
    fen_before:    Optional[str] = None
 
 
@dataclass
class GameData:
    """
    Summary data for an entire game — used to build game narratives.
 
    Attributes:
        white_player    : Name of white player
        black_player    : Name of black player
        opening_name    : Opening played e.g. "Ruy Lopez"
        total_moves     : Total number of moves in the game
        result          : "1-0", "0-1", or "1/2-1/2"
        blunders        : List of MoveData objects that were blunders
        mistakes        : List of MoveData objects that were mistakes
        inaccuracies    : List of MoveData objects that were inaccuracies
        white_accuracy  : White's accuracy percentage (0-100)
        black_accuracy  : Black's accuracy percentage (0-100)
        decisive_moment : The single move_number that swung the game most
    """
    white_player:    str
    black_player:    str
    total_moves:     int
    result:          str
    blunders:        List[MoveData]
    mistakes:        List[MoveData]
    inaccuracies:    List[MoveData]
    white_accuracy:  float          = 0.0
    black_accuracy:  float          = 0.0
    opening_name:    str            = "Unknown Opening"
    decisive_moment: Optional[int]  = None
 
 
# =============================================================================
# HELPER FUNCTIONS — Convert numbers to human-readable words
# =============================================================================
 
def _score_to_words(centipawns: int) -> str:
    """
    Converts a centipawn score into a plain English description.
 
    Centipawns: 100cp = 1 pawn of advantage.
    Positive = White is winning. Negative = Black is winning.
 
    This is critical because you do NOT want the LLM seeing raw numbers.
    "White has a decisive advantage" is far more useful to the LLM than "+450".
 
    Args:
        centipawns : Engine evaluation score.
 
    Returns:
        str: Human-readable description of who is winning and by how much.
 
    Examples:
        _score_to_words(+20)  → "roughly equal position"
        _score_to_words(+120) → "White has a clear advantage"
        _score_to_words(-450) → "Black has a decisive, near-winning advantage"
        _score_to_words(0)    → "completely equal position"
    """
    cp = centipawns  # short alias
 
    # Determine who is winning
    if cp > 0:
        side = "White"
        winning = True
    elif cp < 0:
        side = "Black"
        cp = abs(cp)       # work with absolute value for threshold checks
        winning = True
    else:
        return "a completely equal position"
 
    # Map absolute centipawn value to description
    if cp < 20:
        return "a roughly equal position"
    elif cp < 50:
        return f"{side} has a tiny edge"
    elif cp < 100:
        return f"{side} has a slight advantage"
    elif cp < 200:
        return f"{side} has a clear advantage"
    elif cp < 350:
        return f"{side} has a significant advantage"
    elif cp < 600:
        return f"{side} has a decisive, near-winning advantage"
    else:
        return f"{side} has a completely winning position"
 
 
def _score_delta_to_severity(delta: int) -> str:
    """
    Converts a score DROP (positive number = how much was lost) into
    a severity word for use in prompts.
 
    Args:
        delta : Amount of centipawns lost (always pass as positive number).
                e.g., if score went from +120 to -200, delta = 320
 
    Returns:
        str: Severity word.
 
    Examples:
        _score_delta_to_severity(80)  → "moderate"
        _score_delta_to_severity(200) → "serious"
        _score_delta_to_severity(350) → "severe"
        _score_delta_to_severity(500) → "catastrophic"
    """
    if delta < 75:
        return "minor"
    elif delta < 150:
        return "moderate"
    elif delta < 300:
        return "serious"
    elif delta < 500:
        return "severe"
    else:
        return "catastrophic"
 
 
def _get_phase_context(phase: str) -> str:
    """
    Returns a one-line coaching reminder about what matters in each game phase.
    This is appended to prompts so the LLM frames its explanation in phase context.
 
    Args:
        phase : "opening", "middlegame", or "endgame"
 
    Returns:
        str: A brief phase-specific coaching note.
    """
    contexts = {
        "opening": (
            "In the opening, the priorities are: develop pieces quickly, "
            "control the center, and ensure king safety by castling early."
        ),
        "middlegame": (
            "In the middlegame, tactics and piece coordination are paramount. "
            "Watch for forks, pins, skewers, and undefended pieces."
        ),
        "endgame": (
            "In the endgame, king activity is crucial. Passed pawns become "
            "powerful, and accurate technique determines the outcome."
        ),
    }
    return contexts.get(phase.lower(), contexts["middlegame"])
 
 
def _format_score_change(score_before: int, score_after: int, player_color: str) -> str:
    """
    Creates a clear, formatted description of how the score changed after a move.
 
    Important: Black's moves flip the sign, because when Black plays a good move,
    the score from White's perspective goes down (more negative).
 
    Args:
        score_before : Centipawn score before the move.
        score_after  : Centipawn score after the move.
        player_color : "White" or "Black" — who made the move.
 
    Returns:
        str: Human-readable score change description.
 
    Example:
        _format_score_change(+120, -200, "White")
        → "Position shifted from White has a clear advantage to
           Black has a significant advantage"
    """
    before_words = _score_to_words(score_before)
    after_words  = _score_to_words(score_after)
    return f"Position shifted from '{before_words}' to '{after_words}'"
 
 
# =============================================================================
# CORE PROMPT BUILDERS
# =============================================================================
 
def build_blunder_prompt(data: MoveData) -> str:
    """
    Builds a detailed prompt for explaining a BLUNDER or MISTAKE.
 
    This is your primary function. It is called by BlunderExplainer
    every time a bad move needs to be explained.
 
    Design principles:
    1. Give the LLM ONLY facts from the engine — no room to invent.
    2. Tell the LLM EXACTLY what format to respond in.
    3. Include phase context so the explanation is strategically accurate.
    4. Explicitly say "Do NOT invent lines" to prevent hallucination.
 
    Args:
        data : MoveData object with all engine information about the move.
 
    Returns:
        str: A complete, structured prompt ready to send to any LLM.
 
    Example prompt output (abbreviated):
        "You are a chess coach. A player just made a BLUNDER...
         Move played: e4e5
         Score before: +120cp (White has a clear advantage)
         ..."
    """
    # Calculate how bad the move was
    score_drop = abs(data.score_after - data.score_before)
    severity   = _score_delta_to_severity(score_drop)
    score_desc = _format_score_change(data.score_before, data.score_after, data.player_color)
 
    # Get phase-specific coaching context
    phase_context = _get_phase_context(data.phase)
 
    # Convert raw scores to readable words for the LLM
    before_words = _score_to_words(data.score_before)
    after_words  = _score_to_words(data.score_after)
    best_words   = _score_to_words(data.best_score)
 
    # Build the prompt — carefully structured
    prompt = f"""You are a chess coach explaining a move to a student. Be specific and educational.
 
MOVE INFORMATION (all data is from a chess engine — treat it as ground truth):
  Move number   : {data.move_number}
  Player        : {data.player_color}
  Piece moved   : {data.piece}
  Move played   : {data.move_played}
  Category      : {data.category.upper()} ({severity} error)
  Game phase    : {data.phase}
 
ENGINE EVALUATION:
  Score BEFORE this move : {data.score_before:+d} centipawns ({before_words})
  Score AFTER this move  : {data.score_after:+d} centipawns ({after_words})
  Score drop             : {score_drop} centipawns ({severity})
  {score_desc}
 
BEST MOVE (according to engine):
  Best move was : {data.best_move}
  Score if best : {data.best_score:+d} centipawns ({best_words})
 
PHASE CONTEXT:
  {phase_context}
 
YOUR TASK:
  Write exactly 3 sentences:
  Sentence 1: State what went wrong with {data.move_played} using the score data above.
  Sentence 2: Explain what {data.best_move} achieves that {data.move_played} does not.
  Sentence 3: Give one practical lesson the student should remember.
 
STRICT RULES:
  - Use ONLY the data provided above. Do not invent moves, variations, or reasoning.
  - Do not say "I think" or "probably" — state facts from the engine data.
  - Write for an intermediate chess student. No jargon without brief explanation.
  - Do NOT mention centipawn numbers in your explanation — use the word descriptions.
"""
    return prompt.strip()
 
 
def build_good_move_prompt(data: MoveData) -> str:
    """
    Builds a prompt for explaining why a GOOD or EXCELLENT move was strong.
 
    Used when the player found a strong move — useful for positive reinforcement
    and helping students understand what they did right.
 
    Args:
        data : MoveData with category = "good" or "excellent"
 
    Returns:
        str: Structured prompt for LLM.
    """
    before_words = _score_to_words(data.score_before)
    after_words  = _score_to_words(data.score_after)
    phase_context = _get_phase_context(data.phase)
 
    # For good moves, score improves (or stays equal) — calculate improvement
    score_change = data.score_after - data.score_before
    if data.player_color == "Black":
        score_change = -score_change  # Black benefits when score becomes more negative
 
    improvement = "maintained the advantage" if score_change <= 10 else f"improved the position by {score_change} centipawns"
 
    prompt = f"""You are a chess coach praising a student for a strong move.
 
MOVE INFORMATION:
  Move number   : {data.move_number}
  Player        : {data.player_color}
  Piece moved   : {data.piece}
  Move played   : {data.move_played}
  Category      : {data.category.upper()}
  Game phase    : {data.phase}
 
ENGINE EVALUATION:
  Score BEFORE : {data.score_before:+d} centipawns ({before_words})
  Score AFTER  : {data.score_after:+d} centipawns ({after_words})
  Result       : The move {improvement}
 
PHASE CONTEXT:
  {phase_context}
 
YOUR TASK:
  Write 2 sentences:
  Sentence 1: Explain what makes {data.move_played} a strong choice, using the score data.
  Sentence 2: Explain the strategic or tactical idea behind it.
 
STRICT RULES:
  - Use ONLY the data provided. Do not invent specific tactical lines.
  - Be encouraging but factual.
  - No centipawn numbers in your response — use word descriptions.
"""
    return prompt.strip()
 
 
def build_game_summary_prompt(game: GameData) -> str:
    """
    Builds a prompt asking the LLM to narrate the story of an entire game.
 
    This creates the "NarrativeBuilder" output — a flowing, story-like
    description of how the game unfolded, what went wrong, and the key moment.
 
    Args:
        game : GameData object with full game statistics.
 
    Returns:
        str: Structured prompt for a game narrative.
    """
    # Count total errors
    total_errors = len(game.blunders) + len(game.mistakes) + len(game.inaccuracies)
 
    # Format blunder list for the prompt
    blunder_list = ""
    for i, b in enumerate(game.blunders[:3], 1):  # max 3 blunders to keep prompt short
        blunder_list += (
            f"  Blunder {i}: Move {b.move_number} by {b.player_color} "
            f"({b.move_played}) — score dropped {abs(b.score_after - b.score_before)}cp\n"
        )
    if not blunder_list:
        blunder_list = "  None — both players avoided blunders.\n"
 
    # Result in plain English
    result_map = {
        "1-0":     f"{game.white_player} (White) won",
        "0-1":     f"{game.black_player} (Black) won",
        "1/2-1/2": "the game ended in a draw",
    }
    result_words = result_map.get(game.result, "the game concluded")
 
    prompt = f"""You are a chess commentator writing a post-game analysis narrative.
 
GAME INFORMATION:
  White player    : {game.white_player}
  Black player    : {game.black_player}
  Opening         : {game.opening_name}
  Total moves     : {game.total_moves}
  Result          : {game.result} ({result_words})
 
ACCURACY STATISTICS:
  White accuracy  : {game.white_accuracy:.1f}%
  Black accuracy  : {game.black_accuracy:.1f}%
  Total errors    : {total_errors} (blunders: {len(game.blunders)}, mistakes: {len(game.mistakes)}, inaccuracies: {len(game.inaccuracies)})
 
KEY BLUNDERS:
{blunder_list}
DECISIVE MOMENT:
  {"Move " + str(game.decisive_moment) + " was the turning point of the game." if game.decisive_moment else "No single decisive moment identified."}
 
YOUR TASK:
  Write a 3-paragraph game narrative:
  Paragraph 1 (Opening): How the game started, the opening played, early plans.
  Paragraph 2 (Key Moment): What the critical error was, what changed, how the advantage shifted.
  Paragraph 3 (Conclusion): How the game ended, what both players can learn.
 
STRICT RULES:
  - Use ONLY the statistics and move data provided above.
  - Write in an engaging style, like a chess broadcaster.
  - Do not invent specific move variations or lines.
  - Keep each paragraph to 3-4 sentences.
"""
    return prompt.strip()
 
 
def build_coaching_prompt(
    position_description: str,
    score: int,
    available_moves: List[str],
    player_level: str = "intermediate",
    best_move: Optional[str] = None
) -> str:
    """
    Builds a prompt for interactive ChessCoach mode.
 
    When a student asks "what should I do here?", this prompt
    asks the LLM to explain the position and suggest a plan.
 
    Args:
        position_description : Plain English or FEN description of the position.
        score                : Current engine evaluation in centipawns.
        available_moves      : List of top engine moves to consider.
        player_level         : "beginner" / "intermediate" / "advanced"
        best_move            : The single best move according to engine.
 
    Returns:
        str: Coaching prompt for the LLM.
    """
    score_words = _score_to_words(score)
    moves_str   = ", ".join(available_moves[:5]) if available_moves else "not provided"
 
    # Adjust explanation depth based on player level
    level_instruction = {
        "beginner":     "Use simple language. Explain chess terms when you use them. Focus on one idea.",
        "intermediate": "Use standard chess language. Explain the key strategic and tactical ideas.",
        "advanced":     "Use precise chess terminology. Include nuanced positional reasoning.",
    }.get(player_level.lower(), "Use standard chess language.")
 
    prompt = f"""You are an interactive chess coach helping a student during a game.
 
CURRENT POSITION:
  Position      : {position_description}
  Evaluation    : {score:+d} centipawns ({score_words})
  Best move     : {best_move if best_move else "not specified"}
  Top moves     : {moves_str}
  Student level : {player_level}
 
COACHING INSTRUCTION:
  {level_instruction}
 
YOUR TASK:
  Write 3-4 sentences:
  1. Briefly describe what is happening in the position right now.
  2. Explain the key threat or opportunity the player should focus on.
  3. Suggest a plan (based on the top moves listed above only).
  4. Give one practical tip for this type of position.
 
STRICT RULES:
  - Only suggest moves from the top moves list provided.
  - Do not invent tactics or lines not supported by the data.
  - Be encouraging and clear.
"""
    return prompt.strip()
 
 
def build_lesson_prompt(
    mistake_category: str,
    example_moves: List[MoveData],
    phase: str = "middlegame"
) -> str:
    """
    Builds a prompt asking the LLM to extract a general lesson from
    a pattern of mistakes made throughout the game.
 
    Used by LessonExtractor to turn repeated errors into coaching advice.
 
    Args:
        mistake_category : e.g. "blunder", "mistake", "inaccuracy"
        example_moves    : List of MoveData objects showing the pattern.
        phase            : Which phase these mistakes happened in.
 
    Returns:
        str: Lesson extraction prompt.
    """
    # Build a concise list of example errors
    examples_text = ""
    for i, move in enumerate(example_moves[:4], 1):
        drop = abs(move.score_after - move.score_before)
        examples_text += (
            f"  Example {i}: Move {move.move_number}, {move.move_played} "
            f"by {move.player_color} — lost {drop}cp in the {move.phase}\n"
        )
 
    phase_context = _get_phase_context(phase)
 
    prompt = f"""You are a chess coach identifying a pattern in a student's mistakes.
 
PATTERN DETECTED:
  Mistake type  : {mistake_category.upper()}
  Phase         : {phase}
  Number found  : {len(example_moves)} times in this game
 
EXAMPLES FROM THIS GAME:
{examples_text}
PHASE CONTEXT:
  {phase_context}
 
YOUR TASK:
  Write a 2-sentence lesson:
  Sentence 1: Name the pattern you see in these mistakes and what causes it.
  Sentence 2: Give ONE specific, actionable thing the student should practice.
 
STRICT RULES:
  - Base your lesson ONLY on the examples listed above.
  - Make it concrete and actionable, not vague like "play more carefully".
  - Do not mention specific moves or positions not listed above.
"""
    return prompt.strip()
 
 
# =============================================================================
# PromptBuilder CLASS — Clean wrapper for all the above functions
# =============================================================================
 
class PromptBuilder:
    """
    A class that wraps all prompt-building functions into one clean interface.
 
    WHY USE THIS CLASS instead of calling functions directly?
    - You can configure it once (player level, verbosity) and reuse it.
    - Your BlunderExplainer and MoveExplainer just hold a PromptBuilder instance.
    - Easy to test — just create PromptBuilder() and call its methods.
 
    Example:
        builder = PromptBuilder(player_level="beginner", verbose=False)
 
        data = MoveData(
            move_played="e4e5", score_before=120, score_after=-200,
            best_move="d4", best_score=150, category="blunder",
            piece="pawn", phase="middlegame", move_number=14,
            player_color="White"
        )
 
        prompt = builder.for_blunder(data)
        print(prompt)
    """
 
    def __init__(self, player_level: str = "intermediate", verbose: bool = True):
        """
        Args:
            player_level : Default level for coaching prompts: beginner/intermediate/advanced
            verbose      : If True, prompts include more context. If False, shorter prompts.
        """
        self.player_level = player_level
        self.verbose = verbose
 
    def for_blunder(self, data: MoveData) -> str:
        """Build a prompt for a blunder or mistake."""
        return build_blunder_prompt(data)
 
    def for_good_move(self, data: MoveData) -> str:
        """Build a prompt for a good or excellent move."""
        return build_good_move_prompt(data)
 
    def for_game_summary(self, game: GameData) -> str:
        """Build a prompt for a full game narrative."""
        return build_game_summary_prompt(game)
 
    def for_coaching(self, position_description: str, score: int,
                     available_moves: List[str], best_move: Optional[str] = None) -> str:
        """Build an interactive coaching prompt."""
        return build_coaching_prompt(
            position_description, score, available_moves,
            self.player_level, best_move
        )
 
    def for_lesson(self, mistake_category: str,
                   example_moves: List[MoveData], phase: str = "middlegame") -> str:
        """Build a lesson extraction prompt."""
        return build_lesson_prompt(mistake_category, example_moves, phase)
 
    def score_to_words(self, centipawns: int) -> str:
        """Public access to the score converter utility."""
        return _score_to_words(centipawns)
 
    def score_delta_to_severity(self, delta: int) -> str:
        """Public access to the severity converter utility."""
        return _score_delta_to_severity(delta)
    

 
# =============================================================================
# QUICK TEST — Runing: python prompt_builder.py
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("prompt_builder.py — Quick Test")
    print("=" * 65)
 
    # ── Test 1: Score to words ─────────────────────────────────────────
    print("\n[TEST 1] _score_to_words()")
    test_scores = [0, 20, 80, 150, 300, 500, -120, -400]
    for s in test_scores:
        print(f"  {s:+5d}cp  →  {_score_to_words(s)}")
 
    # ── Test 2: Severity ───────────────────────────────────────────────
    print("\n[TEST 2] _score_delta_to_severity()")
    test_deltas = [50, 100, 200, 350, 600]
    for d in test_deltas:
        print(f"  Drop {d:3d}cp  →  {_score_delta_to_severity(d)}")
 
    # ── Test 3: Build a blunder prompt ────────────────────────────────
    print("\n[TEST 3] build_blunder_prompt()")
    move_data = MoveData(
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
    prompt = build_blunder_prompt(move_data)
    print(prompt)
 
    # ── Test 4: Build a good move prompt ──────────────────────────────
    print("\n" + "=" * 65)
    print("[TEST 4] build_good_move_prompt()")
    good_data = MoveData(
        move_played  = "Nf3",
        score_before = +50,
        score_after  = +130,
        best_move    = "Nf3",
        best_score   = +130,
        category     = "excellent",
        piece        = "knight",
        phase        = "opening",
        move_number  = 5,
        player_color = "White"
    )
    print(build_good_move_prompt(good_data))
 
    # ── Test 5: PromptBuilder class ───────────────────────────────────
    print("\n" + "=" * 65)
    print("[TEST 5] PromptBuilder class")
    builder = PromptBuilder(player_level="beginner")
    print(f"Player level : {builder.player_level}")
    print(f"Score words  : {builder.score_to_words(+250)}")
    print(f"Severity     : {builder.score_delta_to_severity(320)}")
    p = builder.for_blunder(move_data)
    print(f"Prompt length: {len(p)} characters ✓")
 
    print("\n" + "=" * 65)
    print("All tests passed!")
    print("Next: Use PromptBuilder in move_explainer.py (Module 3)")
    print("=" * 65)